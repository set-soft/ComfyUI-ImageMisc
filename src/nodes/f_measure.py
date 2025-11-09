import torch
import numpy as np
import scipy
# Epsilon: small value to avoid "divide by 0" errors
EPS = 1e-8


def get_f_measure(pred: torch.Tensor, gt: torch.Tensor, beta2: float = 0.3) -> float:
    """
    Calculates the maximum F-measure score for a continuous prediction against a binary ground truth.

    The F-measure evaluates the balance between precision and recall. Since the prediction
    is a continuous map (0.0 to 1.0), this function iterates through 256 possible
    thresholds to binarize the prediction. It calculates the F-measure for each
    threshold and returns the highest (max) score found. This provides a fair
    evaluation of the prediction's structural quality, independent of its overall brightness.

    Args:
        pred (torch.Tensor): The continuous prediction mask (normalized to [0, 1]).
        gt (torch.Tensor): The binary ground truth mask (values are 0 or 1).
        beta2 (float): The beta-squared value for the F-measure. The standard value of 0.3
                       is used to weigh precision more heavily than recall. Defaults to 0.3.

    Returns:
        float: The maximum F-measure score found across all thresholds.
    """
    # Initialize f_max to store the highest F-measure score found so far.
    f_max = 0.0
    f = []

    # Iterate through 256 evenly spaced thresholds from 0.0 to 1.0.
    # This corresponds to testing every possible 8-bit grayscale value as the cutoff.
    # The thresholds tensor is created on the same device as the input for efficiency.
    for threshold in torch.linspace(0, 1, 256, device=gt.device):
        # Binarize the continuous prediction map using the current threshold.
        # Pixels >= threshold become 1.0 (positive), and others become 0.0 (negative).
        pred_binary = (pred >= threshold).float()

        # Calculate True Positives (TP): pixels that are positive in both the prediction and ground truth.
        # Element-wise multiplication results in 1 only where both are 1.
        tp = (pred_binary * gt).sum()

        # Optimization: If there are no true positives, the F-measure will be 0.
        # We can skip the rest of the calculations for this threshold.
        if tp == 0:
            f.append((threshold, 0))
            continue

        # Calculate Precision = TP / (TP + FP).
        # The sum of `pred_binary` gives the total number of predicted positives (TP + FP).
        precision = tp / (pred_binary.sum() + EPS)

        # Calculate Recall = TP / (TP + FN).
        # The sum of `gt` gives the total number of actual positives (TP + FN).
        recall = tp / (gt.sum() + EPS)

        # Calculate the F-beta score using the computed precision and recall.
        # The beta^2=0.3 value is standard in saliency detection literature.
        f_beta = (1 + beta2) * precision * recall / (beta2 * precision + recall + EPS)
        f.append((threshold, f_beta))

        # Update f_max if the F-beta score for the current threshold is the highest yet.
        # .item() extracts the single float value from the 0-dimensional tensor.
        if f_beta > f_max:
            f_max = f_beta.item()

    # After checking all thresholds, return the maximum score found.
    return f_max, f


def get_weighted_f_measure(pred: torch.Tensor, gt: torch.Tensor, beta2: float = 0.3) -> float:
    """
    Calculates the true research-grade Weighted F-measure (F_beta^w).

    This is a faithful and optimized port of the reference algorithm from the
    "How to Evaluate Foreground Maps?" paper, ensuring verifiable results.
    The implementation performs pre-checks on the GPU for efficiency before
    transferring data to the CPU for SciPy-based calculations.

    Args:
        pred (torch.Tensor): The continuous prediction mask (normalized to [0, 1]).
        gt (torch.Tensor): The binary ground truth mask (values are 0 or 1).
        beta2 (float): The beta-squared value for the F-measure. Defaults to 0.3.

    Returns:
        float: The final Weighted F-measure score.
    """
    # --- Step 1: GPU-side Pre-checks for Efficiency ---

    # Handle the edge case of an all-black ground truth on the GPU.
    # If the ground truth is empty, the score is 1 minus the mean of the prediction.
    # A perfect prediction (all black) would yield a score of 1.
    # This avoids the expensive CPU transfer and SciPy calculations entirely.
    if torch.mean(gt) == 0.0:
        return (1.0 - pred.mean()).item()

    # --- Step 2: Data Transfer to CPU for NumPy/SciPy Processing ---

    # Move tensors to the CPU and convert to NumPy arrays. SciPy functions require this.
    # Squeezing removes any singleton channel dimensions (e.g., from [1, H, W] to [H, W]).
    gt_np = gt.squeeze().cpu().numpy()
    pred_np = pred.squeeze().cpu().numpy()

    # --- Step 3: Creation of Core Error and Dependency Maps ---

    # Create boolean masks for foreground (gt_mask) and background (not_gt_mask) regions.
    # np.isclose is used for safe floating-point comparison.
    gt_mask = np.isclose(gt_np, 1)
    not_gt_mask = np.logical_not(gt_mask)

    # Calculate the initial, simple absolute pixel-wise error map.
    E = np.abs(pred_np - gt_np)

    # Calculate the Euclidean Distance Transform on the INVERTED mask.
    # For each background pixel, `dist` will be its distance to the nearest foreground pixel.
    # `idx` will store the coordinates of that nearest foreground pixel. This is key for the next step.
    # Note: The original implementation uses scipy.ndimage.morphology.distance_transform_edt
    # which is an alias for scipy.ndimage.distance_transform_edt.
    dist, idx = scipy.ndimage.morphology.distance_transform_edt(not_gt_mask, return_indices=True)

    # --- Step 4: Pixel Dependency Map (Et -> EA -> min_E_EA) ---

    # Create the "Pixel Dependency" map, starting with a copy of the original error.
    Et = np.array(E)

    # This is the crucial step for pixel dependency. For every background pixel,
    # its error value is replaced with the error value of its NEAREST foreground pixel.
    # This propagates the error from the foreground edges into the background.
    Et[not_gt_mask] = E[idx[0, not_gt_mask], idx[1, not_gt_mask]]

    # Smooth the dependency-aware error map with a Gaussian filter.
    # This creates a soft, blurred error field around the object.
    sigma = 5.0
    EA = scipy.ndimage.gaussian_filter(Et, sigma=sigma, truncate=3 / sigma, mode='constant', cval=0.0)

    # The final error for any FOREGROUND pixel is the MINIMUM of its original error (E)
    # and the new smoothed, dependency-aware error (EA).
    # This prevents unfairly penalizing small errors right at the boundary of a correct prediction.
    # The `where=gt_mask` argument ensures this operation only applies to the foreground.
    min_E_EA = np.minimum(E, EA, where=gt_mask, out=np.array(E))

    # --- Step 5: Pixel Importance Map (B) ---

    # Create the "Pixel Importance" map, which starts as a uniform map of ones.
    B = np.ones(gt_np.shape)

    # For each BACKGROUND pixel, assign an importance weight based on its distance
    # from the foreground. The weight is calculated using a non-linear function
    # that decreases as the distance increases. This makes errors near the object more important.
    B[not_gt_mask] = 2 - np.exp(np.log(1 - 0.5) / 5 * dist[not_gt_mask])

    # The final Weighted Error Map is the element-wise product of the dependency-aware
    # error map and the pixel importance map.
    Ew = min_E_EA * B

    # --- Step 6: Final Metric Computation ---

    # Get a small machine epsilon value for numerically stable division.
    eps = np.spacing(1)

    # Calculate Weighted True Positives (TPw) and False Positives (FPw) from the Ew map.
    # TPw = Sum of foreground weights - Sum of weighted errors in the foreground.
    # FPw = Sum of weighted errors in the background.
    TPw = np.sum(gt_np) - np.sum(Ew[gt_mask])
    FPw = np.sum(Ew[not_gt_mask])

    # Calculate Weighted Recall (R) and Weighted Precision (P).
    # These definitions are specific to this metric's formulation.
    R = 1 - np.mean(Ew[gt_mask])  # Weighed Recall
    P = TPw / (eps + TPw + FPw)  # Weighted Precision

    # The final Weighted F-measure (Q) is calculated using the standard formula
    # with the newly computed weighted Precision and Recall.
    # beta2 = 0.3 is standard, weighing precision more heavily than recall.
    # Q = 2 * (R * P) / (eps + R + P)  # Beta=1
    Q = (1 + beta2) * (R * P) / (eps + R + (beta2 * P))

    # Raise an error if the result is Not a Number (NaN), indicating a potential issue.
    if np.isnan(Q):
        raise ValueError("Weighted F-measure resulted in NaN")

    return Q
