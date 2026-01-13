import torch
import torch.nn.functional as F

def compute_hsb(logits_original, logits_counterfactual):
    """
    Computes Hallucination Sensitivity Bound (HSB) using KL Divergence.
    Formula: KL( P(y|E) || P(y|E') )
    
    Args:
        logits_original: Logits from model run with Evidence E
        logits_counterfactual: Logits from model run with Evidence E'
    """
    # Convert logits to probabilities (Softmax)
    probs_p = F.softmax(logits_original, dim=-1)
    probs_q = F.softmax(logits_counterfactual, dim=-1)

    # Compute KL Divergence
    # kl_div expects log_probabilities for the input (target distribution)
    # Note: PyTorch kl_div argument order is (input, target) where input is log-probs
    log_probs_p = F.log_softmax(logits_original, dim=-1)
    
    # We sum over the vocabulary dimension (last dim) and mean over the sequence
    kl_divergence = F.kl_div(
        F.log_softmax(logits_counterfactual, dim=-1), # Q (approx)
        probs_p,                                      # P (true)
        reduction='batchmean'
    )
    
    return kl_divergence.item()