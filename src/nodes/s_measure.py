import torch
# Epsilon: small value to avoid "divide by 0" errors
EPS = 1e-8


def get_s_measure(pred: torch.Tensor, gt: torch.Tensor, alpha: float = 0.5) -> float:
    """
    Calculates the S-measure for a given prediction and ground truth mask.
    The S-measure evaluates structural similarity, combining object-aware and
    region-aware metrics.

    Args:
        pred (torch.Tensor): The continuous prediction mask (normalized to [0, 1]).
        gt (torch.Tensor): The binary ground truth mask (values are 0 or 1).
        alpha (float): The weight for balancing object-aware vs. region-aware scores.
                       Defaults to 0.5.

    Returns:
        float: The final S-measure score.
    """
    # gt is assumed to be binary
    y = gt.mean()
    if y == 0:
        # If the ground truth is all black, the score is the inverse of the prediction's average.
        # A perfect prediction would also be all black (mean=0), yielding a score of 1.
        x = pred.mean()
        q = 1.0 - x
    elif y == 1:
        # If the ground truth is all white, the score is simply the prediction's average.
        # A perfect prediction would be all white (mean=1), yielding a score of 1.
        x = pred.mean()
        q = x
    else:
        # For a mixed ground truth, balance the object and region scores.
        q = alpha * s_object(pred, gt) + (1 - alpha) * s_region(pred, gt)
        # Ensure the score is non-negative.
        if q < 0:
            return 0

    return q.item()


def s_object(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    """
    Calculates the object-aware structural similarity score.
    This measures the similarity for foreground and background regions separately
    and combines them based on the foreground's size.
    """
    # Create a map of foreground-only predictions
    fg = torch.where(gt == 0, torch.zeros_like(pred), pred)
    # Create a map of background-only predictions (inverted)
    bg = torch.where(gt == 1, torch.zeros_like(pred), 1 - pred)

    # Calculate scores for each region
    o_fg = s_object_calc(fg, gt)
    o_bg = s_object_calc(bg, 1 - gt)

    # Combine scores based on foreground area
    u = gt.mean()  # The ratio of foreground pixels
    q = u * o_fg + (1 - u) * o_bg
    return q


def s_object_calc(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    """
    Helper function to compute the similarity score for a given region (FG or BG).
    """
    # Select the prediction pixels corresponding to the region of interest in the ground truth
    region_pixels = pred[gt == 1]

    # If the region is empty, the score is undefined, but 0 is a safe return.
    if region_pixels.numel() == 0:
        return torch.tensor(0.0, device=pred.device)

    x = region_pixels.mean()

    # Note: The original paper's formula uses variance (sigma**2), but many public
    # implementations use std dev (sigma). We follow this implementation's logic.
    sigma_x = region_pixels.std()

    # A score that rewards high mean (x) but penalizes high variance (sigma_x).
    # It is maximized when x=1 and sigma_x=0.
    score = 2.0 * x / (x**2 + 1.0 + sigma_x + EPS)
    return score


def s_region(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    """
    Calculates the region-aware structural similarity score.
    This divides the image into four quadrants based on the ground truth's
    centroid and computes a weighted SSIM score.
    """
    # Find the centroid (center of mass) of the ground truth mask
    X, Y = s_centroid(gt)

    # Divide the ground truth and prediction into four quadrants based on the centroid
    gt_parts = s_divide_tensor(gt, X, Y)
    pred_parts = s_divide_tensor(pred, X, Y)

    # Calculate the area weights for each quadrant
    w = s_calculate_weights(gt.shape, X, Y)

    # Calculate SSIM for each quadrant and combine the quadrant scores using their area weights
    return sum([w[i] * s_ssim(pred_parts[i], gt_parts[i]) for i in range(4)])


def s_centroid(gt: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Calculates the centroid (center of mass) of a binary mask.
    """
    rows, cols = gt.shape[-2:]
    total = gt.sum()

    # Handle the edge case of an all-black mask
    if total == 0:
        X = torch.tensor(round(cols / 2), device=gt.device)
        Y = torch.tensor(round(rows / 2), device=gt.device)
    else:
        # Create coordinate ranges directly on the target device, avoiding CPU-GPU transfer.
        i_coords = torch.arange(cols, device=gt.device, dtype=gt.dtype)
        j_coords = torch.arange(rows, device=gt.device, dtype=gt.dtype)

        # Calculate weighted average of coordinates
        X = torch.round((gt.sum(dim=0) * i_coords).sum() / total)
        Y = torch.round((gt.sum(dim=1) * j_coords).sum() / total)

    return X.long(), Y.long()


def s_divide_tensor(tensor: torch.Tensor, X: torch.Tensor, Y: torch.Tensor) -> tuple:
    """
    Divides a tensor into four quadrants based on a pivot point (X, Y).
    This is memory-efficient as it returns views, not copies.
    """
    h, w = tensor.shape[-2:]
    LT = tensor[..., :Y, :X]
    RT = tensor[..., :Y, X:]
    LB = tensor[..., Y:, :X]
    RB = tensor[..., Y:, X:]
    return LT, RT, LB, RB


def s_calculate_weights(shape: tuple, X: torch.Tensor, Y: torch.Tensor) -> tuple:
    """Calculates the proportional area of the four quadrants."""
    h, w = shape[-2:]
    area = h * w

    # Ensure coordinates are float for division
    Xf = X.float()
    Yf = Y.float()

    w1 = Xf * Yf / area
    w2 = (w - Xf) * Yf / area
    w3 = Xf * (h - Yf) / area
    w4 = 1.0 - w1 - w2 - w3   # More stable calculation for the last weight

    return (w1, w2, w3, w4)


def s_ssim(pred: torch.Tensor, gt: torch.Tensor) -> float:
    """
    Computes a custom structural similarity (SSIM-like) score between two tensors.
    """
    # If a quadrant is empty, its contribution to similarity is ambiguous.
    # Returning 0 is a safe choice, but 1 could also be argued if gt is also empty.
    if pred.numel() == 0 or gt.numel() == 0:
        return 0.0

    h, w = pred.shape[-2:]
    N = h * w

    # Means
    x = pred.mean()
    y = gt.mean()

    # Variances and Covariance (using unbiased estimator N-1)
    # This is numerically safer than calculating std dev separately.
    sigma_x2 = ((pred - x) * (pred - x)).sum() / (N - 1 + EPS)
    sigma_y2 = ((gt - y) * (gt - y)).sum() / (N - 1 + EPS)
    sigma_xy = ((pred - x) * (gt - y)).sum() / (N - 1 + EPS)

    # Numerator and denominator of the SSIM formula
    alpha = 4 * x * y * sigma_xy
    beta = (x*x + y*y) * (sigma_x2 + sigma_y2)

    # Handle special cases for stability, as defined in the original code
    if alpha != 0:
        Q = alpha / (beta + EPS)
    elif alpha == 0 and beta == 0:
        # If both inputs are flat and identical, similarity is perfect.
        Q = 1.0
    else:
        # If numerator is 0 but denominator isn't, similarity is 0.
        Q = 0

    return Q
