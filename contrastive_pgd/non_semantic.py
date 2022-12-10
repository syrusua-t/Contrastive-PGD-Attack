import torch
import numpy as np
import random
def non_semantic(wavfile: torch.Tensor) -> torch.Tensor: 
    '''
    input:
        [ 1 * seq_len ] 
        wavfile: input wavfile
    output:
        [ 1 * seq_len ]
        non_seman_neg: a non-semantic negative sample
    '''
    d = int(np.random.uniform(8, 12, 1))
    # extract wavfile: originally 2d
    wavfile = wavfile[0]
    # expected length for each segement
    exp_seg_len = len(wavfile) / d
    # keep track of segemented data
    segments = []

    # segmenting the wavfile by sampling lengths from normal distribution
    curEnd = 0
    for i in range(d - 1):
        tmp_len = int(np.random.normal(exp_seg_len, exp_seg_len ** 0.15, 1))
        segments.append(wavfile[curEnd : curEnd + tmp_len])
        curEnd += tmp_len
    segments.append(wavfile[curEnd:])

    # shuffle wavfile segments
    random.shuffle(segments)
    # concat shuffled segments
    non_seman_neg = torch.Tensor([list(torch.cat(segments))])
    return non_seman_neg