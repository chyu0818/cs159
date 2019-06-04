import math
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
import torch.optim as optim
import numpy as np
from torch.autograd import Variable

from ctextgen.dataset import SST_Dataset
from ctextgen.model import RNN_VAE

import argparse
import random
import time


parser = argparse.ArgumentParser(
    description='Conditional Text Generation'
)

parser.add_argument('--gpu', default=False, action='store_true',
                    help='whether to run in the GPU')
parser.add_argument('--model', default='ctextgen', metavar='',
                    help='choose the model: {`vae`, `ctextgen`}, (default: `ctextgen`)')
parser.add_argument('--path', default='saved_models/emotions_7_tfidf_wordvec.bin',
                    metavar='', help='choose the model: from saved_models, (default: `baseline_vae`)')
parser.add_argument('--num_sentences', default='10')

args = parser.parse_args()


mb_size = 32
z_dim = 20
h_dim = 64
lr = 1e-3
lr_decay_every = 1000000
n_iter = 20000
log_interval = 1000
z_dim = h_dim
c_dim = 7

dataset = SST_Dataset()

torch.manual_seed(int(time.time()))

model = RNN_VAE(
    dataset.n_vocab, h_dim, z_dim, c_dim, p_word_dropout=0.3,
    pretrained_embeddings=dataset.get_vocab_vectors(), freeze_embeddings=True,
    gpu=args.gpu
)

if args.gpu:
    model.load_state_dict(torch.load('{}'.format(args.path)))
else:
    model.load_state_dict(torch.load('{}'.format(args.path), map_location=lambda storage, loc: storage))


# go through `num_sentences`
for i in range(int(args.num_sentences)):

    # sample z, c prior
    z = model.sample_z_prior(1)
    c = model.sample_c_prior(1)
    print(c_dim)

    _, c_idx = torch.max(c, dim=1)
    sample_idxs = model.sample_sentence(z, c, temp=0.1)

    # print('\nSentiment: {}'.format(dataset.idx2label(int(c_idx))))
    print("SENTIMENT IS")
    print(c)
    print('Generated: {}'.format(dataset.idxs2sentence(sample_idxs)))










