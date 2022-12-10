import torch
import numpy as np
import random

def rir_augment(audio_vector, room_dim=[10, 7.5, 3.5], rt60_tgt = 0.3):
  
    '''
    INPUT: 

    augment flac file with room impulse response given following arguments:

    -----
    path_to_flac: path to Librispeech flac file 

    -----
    room dimension: a three element int list that describes the room 
    e.g. [length, width, height]

    -----
    rt60_tgt: reverberation time, in seconds

    OUTPUT: 

    Save the augmented flac file in the original data path, with name added sufix
    -rir, return nothing
    '''
    import os
    import pyroomacoustics as pra
    from scipy.io import wavfile
    #generate RIR augmented flac
    fs = 16000

    e_absorption, max_order = pra.inverse_sabine(rt60_tgt, room_dim)
    room = pra.ShoeBox(
        room_dim, fs=fs, materials=pra.Material(e_absorption), max_order=max_order,
        use_rand_ism = True
    )

    # place the source in the room
    room.add_source([2.5, 3.73, 1.76], signal=audio_vector, delay=0.5)

    # define the locations of the microphones
    mic_locs = np.c_[
        [6.3, 4.87, 1.2], [6.3, 4.93, 1.2],  # mic 1  # mic 2
    ]

    room.add_microphone_array(mic_locs)
    room.simulate()

    room.mic_array.to_wav(
        f"tmp.wav",
        norm=True,
        bitdepth=np.float32,
        mono=True
    )

    _, rir_audio = wavfile.read("tmp.wav")

    os.remove("tmp.wav") 

    return torch.tensor(np.array([rir_audio]))
