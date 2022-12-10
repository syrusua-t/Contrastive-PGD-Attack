# Contrastive-PGD-Attack
Contrastive PGD Attack, by Yifu Cai, Hao wang, Yunqing Yu

Project Abstract: 

Modern Automatic Speech Recognition (ASR) architectures often underwent self-supervised learning during the pre-training phase for purpose of learning more robust models with higher accuracy.Previous studies conducted by Raphael Olivier, Hadi Abdullah, and Bhiksha Raj have showcased models pre-trained with self-supervised learning are vulnerable to targeted, transferable adversarial attack[ 5]. This discovery raises alarm among the science community and shall be dealt with before such ASR models are put into production.

In the current literature, the experiments are either not conducted under an over-the-air setting, where crafted audio samples are directly fed into the ASR models without external noise, or conducted under an over-the-air setting but assuming a white-box attack, meaning that the target model is known when generating the adversarial examples. Therefore, further research on whether the same, transferable adversarial attack on ASR models remain effective under an over-the-air setting through zero knowledge of the attacked model is necessary. To the best of our knowledge, no previous research has been conducted on this topic, so we decide to further this investigation through this research.

In the final experiment, we aim to compute our adversarial examples through, still, optimizing an objective function. The original baseline objective function is to add a small perturbation to the input data that maximizes the model loss. We modified our objective function in the final experiment. The objective function is a loss function that joints loss from clean audio, loss from Room Impulse Response (RIR) audio augmented impulse, and contrastive loss between clean audio and RIR-augmented audio.

To replicate the experiment, you will need several things 

Our attack is supported by Robust Speech, credit to Raphael Olivier, who made this amazing repo. You will first need clone the repo and install the required package. 

1. Clone the following repo https://github.com/RaphaelOlivier/robust_speech. Note that this it should be on your python site-package folder

2. You should also install our requirement.txt, which includes additional package required to run our code

3. You will have to download a Wav2Vec2.0 model yourself following the guideline of robust_speech to generate and evaluate adversial examples. The model is too large to be uploaded here. 

4. Once you have a working Wav2Vec2.0 model compatible with robust_speech, which takes a slight training process, you are good to replicate our experiment. 

5. You should place the ucontrastive_pgd folder inside site-package of your python 

6. You should replace the pgd attack site-packages/robust_speech/robust_speech/adversarial/attacks/pgd.py with our pgd file within pgd attack folder. This is a file that contains our new loss function. 
