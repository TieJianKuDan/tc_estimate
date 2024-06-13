import torch
from torch.nn import functional as F


def build_cosin_similarity(emb_feats):
    B,T,c,h,w = emb_feats.shape
    emb_feats = emb_feats.permute(0,1,3,4,2) #  (B,T,h,w,c)
    normalize_feats = emb_feats / (torch.norm(emb_feats,dim=-1,keepdim=True)+1e-6) #  (B,T,h,w,c)
    prev_frame = normalize_feats[:,:T-1].reshape(-1,h*w,c) # (B*(T-1),h*w,c)
    next_frame = normalize_feats[:,1:].reshape(-1,h*w,c,) # (B*(T-1),h*w,c)
    similarity_matrix = torch.bmm(prev_frame,next_frame.permute(0,2,1)).reshape(B,T-1,h*w,h*w) # (N*(T-1)*h*w)
    return similarity_matrix


def bmc_loss(pred, target, noise_var):
    """Compute the Balanced MSE Loss (BMC) between `pred` and the ground truth `targets`.
    Args:
      pred: A float tensor of size [batch, 1].
      target: A float tensor of size [batch, 1].
      noise_var: A float number or tensor.
    Returns:
      loss: A float tensor. Balanced MSE Loss.
    """
    logits = - (pred - target.T).pow(2) / (2 * noise_var)   # logit size: [batch, batch]
    loss = F.cross_entropy(logits, torch.arange(pred.shape[0]))     # contrastive-like loss
    loss = loss * (2 * noise_var).detach()  # optional: restore the loss scale, 'detach' when noise is learnable 

    return loss