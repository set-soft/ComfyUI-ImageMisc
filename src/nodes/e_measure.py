import torch
from typing import Tuple
from . import EPS, F_POINTS


def get_e_measure(
    pred: torch.Tensor,
    gt: torch.Tensor,
    num_thresholds: int = F_POINTS,
    chunk_size: int = -1
) -> Tuple[float, float, float, torch.Tensor, torch.Tensor]:
    """
    Calculates the E-measure scores using a memory-efficient chunking strategy.

    This implementation is fully vectorized within chunks to maintain high performance
    while ensuring that memory usage remains low and predictable, making it suitable
    for high-resolution images.

    Args:
        pred (torch.Tensor): The continuous prediction mask (normalized to [0, 1]).
        gt (torch.Tensor): The binary ground truth mask (values are 0 or 1).
        num_thresholds (int): The number of thresholds to evaluate. Defaults to F_POINTS.
        chunk_size (int): The number of thresholds to process in a single batch.
                          Lower this value if you encounter VRAM issues. Defaults to 16.

    Returns:
        A tuple containing:
        - float: The mean E-measure score across all thresholds.
        - float: The maximum E-measure score across all thresholds.
        - float: The adaptive E-measure score.
        - torch.Tensor: A 1D tensor with the E-measure score for each threshold.
        - torch.Tensor: A 1D tensor with the thresholds.
    """

    # 1. --- Calculate scores for all thresholds using chunking ---

    # Create a 1D tensor of thresholds from 0 to almost 1
    thlist = torch.linspace(0, 1 - 1e-10, num_thresholds, device=pred.device)
    all_scores = []

    if chunk_size == -1:
        chunk_size = round(64 / (pred.shape[0] * pred.shape[1] / (1<<20)))
        chunk_size = min(max(1, chunk_size), num_thresholds)

    # Process thresholds in memory-efficient chunks
    for i in range(0, num_thresholds, chunk_size):
        # Get the current chunk of thresholds
        th_chunk = thlist[i:i + chunk_size]

        # Vectorized operation on the smaller chunk.
        # Reshape pred to (1, H, W) and thlist to (N, 1, 1).
        # Broadcasting (>=) creates a binarized prediction for each threshold.
        # The result `binarized_preds` has a shape of (chunk_size, H, W).
        binarized_preds_chunk = (pred.unsqueeze(0) >= th_chunk.view(-1, 1, 1)).to(pred.dtype)

        # Calculate scores for the current chunk
        scores_chunk = e_calculate_enhanced_scores(binarized_preds_chunk, gt)
        all_scores.append(scores_chunk)

    # Combine the scores from all chunks
    scores = torch.cat(all_scores)

    # 2. --- Calculate the single adaptive score  ---

    # Calculate the adaptive threshold, clamping at 1.0 to be safe.
    adaptive_th = torch.clamp(2 * pred.mean(), max=1.0)
    # Binarize the prediction with this single threshold
    adaptive_pred_binarized = (pred >= adaptive_th).to(pred.dtype)
    # Reuse the same helper function by adding a temporary batch dimension
    adaptive_score_tensor = e_calculate_enhanced_scores(adaptive_pred_binarized.unsqueeze(0), gt)

    # 3. --- Return the final results ---

    return (
        scores.mean().item(),
        scores.max().item(),
        adaptive_score_tensor.item(),
        scores.cpu(), thlist
    )


def e_calculate_enhanced_scores(binarized_preds: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    """
    Helper function to compute E-measure scores for a batch of binarized predictions.
    This function is fully vectorized.

    Args:
        binarized_preds (torch.Tensor): A tensor of binarized predictions, with shape
                                        (N, H, W), where N is the number of thresholds.
        gt (torch.Tensor): The single binary ground truth mask, with shape (H, W).

    Returns:
        torch.Tensor: A 1D tensor of shape (N,) containing the E-measure score
                      for each binarized prediction.
    """
    # Handle the edge case where the ground truth is all black
    if torch.mean(gt) == 0.0:
        # The score is based on how much of the prediction is also black.
        # Original formula: sum(1 - y_pred_th) / (numel - 1)
        # This is equivalent to numel * (1 - mean) / (numel - 1)
        enhanced = 1 - binarized_preds

    # Handle the edge case where the ground truth is all white
    elif torch.mean(gt) == 1.0:
        # The score is based on how much of the prediction is also white.
        # Original formula: sum(y_pred_th) / (numel - 1)
        enhanced = binarized_preds

    # Normal case with a mixed ground truth
    else:
        # Demean the ground truth. `gt_demeaned` has shape (H, W).
        gt_demeaned = gt - gt.mean()

        # Demean the binarized predictions.
        # `mean` is calculated over spatial dims (H, W), keeping the threshold dim.
        # `fm` (foreground map) will have shape (N, H, W).
        fm = binarized_preds - binarized_preds.mean(dim=[-2, -1], keepdim=True)

        # The demeaned GT will be broadcasted to match the shape of `fm`.
        align_matrix = 2 * gt_demeaned * fm / (gt_demeaned.square() + fm.square() + EPS)
        enhanced = (align_matrix + 1).square() / 4

    # Calculate the final score for each threshold by summing over the spatial dimensions.
    # The denominator (y.numel() - 1) is a quirk from the original paper's code.
    scores = torch.sum(enhanced, dim=[-2, -1]) / (gt.numel() - 1 + EPS)

    return scores
