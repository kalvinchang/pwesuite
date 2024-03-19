#!/usr/bin/bash

mkdir computed
touch computed/embd_bpemb.pkl

for LANG in 'en' 'am' 'bn' 'uz' 'pl' 'es' 'sw' 'fr' 'de'; do
  python models/language_modeling/train_apply.py \
                --data \"data/multi.tsv\" \
                --lang ${LANG} \
                --output \"computed/tmp/masked_lm_panphon_${LANG}.pkl\"
done;

# TODO: unpickle the embeddings
# TODO: concatenate the embeddings into computed/embd_masked_lm.pkl (must match order of multi.tsv)
