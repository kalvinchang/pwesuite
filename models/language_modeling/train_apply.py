#!/usr/bin/env python3
import sys
import random
import torch
import argparse
import os
import yaml


args = argparse.ArgumentParser()
args.add_argument("-d", "--data", default="data/multi.tsv")
# lang=all training multilingually and is not yet supported by this script
args.add_argument("-l", "--lang", default="en")
args.add_argument("-o", "--output", default="computed/embd_bpemb.pkl")
args.add_argument(
    "-nk", "--number-thousands", type=int, default=200,
    help="Number of training data to use (in thousands) for training",
)
args.add_argument(
    "--eval-train-full", action="store_true",
    help="Compute correlations also for full the training data instead of just 1k sample. This will be significantly slower."
)
args.add_argument("--features", default="panphon")
args.add_argument("--dimension", type=int, default=300)
args = args.parse_args()

with open('models/language_modeling/hyperparams.yaml', 'r') as f:
    hyperparams = yaml.load(f, yaml.CLoader)

vocab_file = f"data/vocab/ipa_{args.lang}.txt"
os.system(f'''
    python3 ./models/language_modeling/train_masked_lm.py --batch_size={hyperparams['batch_size']} \
        --classifier_dropout={hyperparams['classifier_dropout']} \
        --dim_feedforward={hyperparams['dim_feedforward']} \
        --dropout={hyperparams['dropout']} \
        --embedding_dim={args.dimension} \
        --epochs={hyperparams['epochs']} \
        --lang_codes={args.lang} \
        --lr={hyperparams['lr']} \
        --mask_percent={hyperparams['mask_percent']} \
        --num_heads={hyperparams['num_heads']} \
        --num_layers={hyperparams['num_layers']} \
        --sweeping=true \
        --vocab_file={vocab_file} \
        --wandb_entity=llab-reconstruction \
        --wandb_name= \
        --output={args.output}
    ''')
