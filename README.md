# Contrastive-PGD-Attack
Contrastive PGD Attack, by Yifu Cai, Hao Wang, Yunqing Yu

Project Abstract: 

Modern Automatic Speech Recognition (ASR) architectures often underwent self-supervised learning during the pre-training phase for purpose of learning more robust models with higher accuracy.Previous studies conducted by Raphael Olivier, Hadi Abdullah, and Bhiksha Raj have showcased models pre-trained with self-supervised learning are vulnerable to targeted, transferable adversarial attack[ 5]. This discovery raises alarm among the science community and shall be dealt with before such ASR models are put into production.

In the current literature, the experiments are either not conducted under an over-the-air setting, where crafted audio samples are directly fed into the ASR models without external noise, or conducted under an over-the-air setting but assuming a white-box attack, meaning that the target model is known when generating the adversarial examples. Therefore, further research on whether the same, transferable adversarial attack on ASR models remain effective under an over-the-air setting through zero knowledge of the attacked model is necessary. To the best of our knowledge, no previous research has been conducted on this topic, so we decide to further this investigation through this research.

In the final experiment, we aim to compute our adversarial examples through, still, optimizing an objective function. The original baseline objective function is to add a small perturbation to the input data that maximizes the model loss. We modified our objective function in the final experiment. The objective function is a loss function that joints loss from clean audio, loss from Room Impulse Response (RIR) audio augmented impulse, and contrastive loss between clean audio and RIR-augmented audio.

To replicate the experiment, you will need several things 

Our attack is supported by Robust Speech, credit to Raphael Olivier, who made this amazing repo. You will first need clone the repo and install the required package. 

1. Clone the following repo https://github.com/RaphaelOlivier/robust_speech. Note that this it should be on your python site-package folder
```
git clone https://github.com/RaphaelOlivier/robust_speech.git
cd robust_speech
pip install .
```

2. You should also double check against our requirement.txt and install additional packages required to run our code.

3. You need to create the following root structure and download librispeech dataset to the data folder.
```
root
│
└───data
│   │ # where datasets are dumped (e.g. download LibriSpeech here)
│
└───models
│   │
│   └───model_name1
│   │   │   model.ckpt
│   
└───tokenizers   
│   │ # where all tokenizers are saved
│   
└───trainings
│   │  # where your custom models are trained
│  
└───attacks
|   |
│   └───attack_name
│   │   │
│   │   └───1234 # seed
│   │   │   │
│   │   │   └───model_name1
│   │   │   │   │ # your attack results
```

3. Before performing the attack, replace pgd.py in ./site-packages/robust_speech/adversarial/attacks with our pdg.py file and add our contrastive_pgd folder to the directory. This change is done to update the old loss function for pgd attack with our new loss function with contrastive learning.

4. You will have to download a Wav2Vec2.0 model yourself following the guideline of robust_speech to generate and evaluate adversial examples. First, we need to load the correct noise-robust model from huggingface. To do this, go to ./robust_speech/recipes/train_configs/LibpriSpeech and add in our wav2vec2-large-robust-ft-swbd-300h.yaml. Second, to generate the attack, go to ./robust_speech/recipes/LibriSpeech/pgd and duplicate w2v2_large_960h.yaml. Rename the duplicated file as my_attack.yaml and change model_name under model information to wav2vec2-large-robust-ft-swbd-300h.

5. Generating the attack for wav2vec2 is quite tricky. The tokenizer for HuggingFace wav2vec2 is not directly compatible to the robust_speech package. Therefore, it is necessary to first generate a tokenizer from the data, then do a slight retraining of the final linear layer in wav2vec2. This can be done with the following command: 
```
# in ./recipes/

# This will train a model first
python train_model.py train_configs/LibriSpeech/wav2vec2-large-robust-ft-swbd-300h.yaml --root=/path/to/results/folder
mv /path/to/training/outputs/folder/*.ckpt /path/to/models/folder/wav2vec2-large-robust-ft-swbd-300h/
python evaluate.py attack_configs/LibriSpeech/pgd/my_attack.yaml --root=/path/to/results/folder --snr=30
```

6. With the attack generated, you can directly use the Hosted inference API for Data2Vec on https://huggingface.co/facebook/data2vec-audio-large-960h to evaluate the effectiveness of the attack.
