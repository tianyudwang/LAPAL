# Latent Policies for Adversarial Imitation Learning (LAPAL)
This is the code implementation of the paper "Latent Policies for Adversarial Imitation Learning".
- [Project website](https://tianyudwang.github.io/lapal/)
- [Paper](https://arxiv.org/abs/2206.11299)


## Installation
1. Install dependencies
```bash
pip install -r requirements.txt
```
2. Install modified stable-baselines3 from source
```bash
cd stable-baselines3 
pip install -e .
```
3. Install LAPAL
```bash
cd ..
pip install -e .
```

## Training
```bash
python3 gcl/scripts/train_lapal.py configs/halfcheetah/LAPAL.yml
```

