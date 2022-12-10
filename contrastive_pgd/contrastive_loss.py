import torch
import numpy as np
import random

def contrastive_loss(c_raw, c_rir, c_n, c_ns, t=0.07):
    '''
    Calculate for the contrastive loss between c_raw, c_noisy, c_n, c_ns

    Input: (batch_size, time_steps, feat_size=1000)
    c_raw   -> prediction from raw batch
    c_rir -> prediction from rir batch
    c_n     -> prediction from negative batch
    c_ns    -> prediction from non-semantic batch

    Output:
    loss -> contrastive loss
    '''
    c_raw, c_rir, c_n, c_ns = c_raw.to('cuda'), c_rir.to('cuda'), c_n.to('cuda'), c_ns.to('cuda')
    c_raw, c_rir, c_n, c_ns = c_raw[0].flatten(), c_rir[0].flatten(), c_n[0].flatten(), c_ns[0].flatten() # (time_steps, feat_size)
    # Process input length - flatten probabilities & shortten to same length
    min_length = min(len(c_raw), len(c_rir), len(c_n), len(c_ns))
    c_raw, c_rir, c_n, c_ns = c_raw[:min_length], c_rir[:min_length], c_n[:min_length], c_ns[:min_length]
    # Calculate CosineSimilarity
    cos = torch.nn.CosineSimilarity(dim=0, eps=1e-6)
    cos_rir_raw = torch.exp(cos(c_rir, c_raw)/t)
    cos_rir_n = torch.exp(cos(c_rir, c_n)/t)
    cos_rir_ns = torch.exp(cos(c_rir, c_ns)/t)
    loss = -torch.log(cos_rir_raw/(cos_rir_raw+cos_rir_n+cos_rir_ns))
    return loss