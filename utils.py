import torch

def token_accuracy(result: torch.Tensor, target: torch.Tensor, mask_token_mask: torch.Tensor):
    """Calculate MLM accuracy between ONLY masked words
    Args:
        result: result calculated by model
        target: real target
        inverse_token_mask: well-known inverse token mask
    Returns:
        MLM accuracy
    """
    r = result.argmax(-1).masked_select(mask_token_mask)
    t = target.masked_select(mask_token_mask)
    s = (r == t).sum()
    return round(float(s / mask_token_mask.sum()), 2)