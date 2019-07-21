# Emotion-guided text generation with VAE
VAE architectures have been recently very successful at realistic text generation. In the standard VAE architecture, the prior for the latent variable is taken from a normal $N(0, 1)$ distribution. In this paper, an emotion guided GMM is introduced as the prior for a structured latent variable $\mathbf{c}$. The effects of introducing different GMM priors are analyzed, as well as the control they give over the final text generation.

This repo contains all code used to train models, as well as results. To run the model (after all needed dependencies are installed), run `python train_vae.py`
