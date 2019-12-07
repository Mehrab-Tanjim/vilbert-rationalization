# import stuff
# %load_ext autoreload
# %autoreload 2
# %matplotlib inline

from random import randint

import numpy as np
import torch

# Load model
from models import InferSent
model_version = 1
MODEL_PATH = "save/encoder/infersent%s.pkl" % model_version
params_model = {'bsize': 64, 'word_emb_dim': 300, 'enc_lstm_dim': 2048,
                'pool_type': 'max', 'dpout_model': 0.0, 'version': model_version}
model = InferSent(params_model)
model.load_state_dict(torch.load(MODEL_PATH))

# Keep it on CPU or put it on GPU
use_cuda = False
model = model.cuda() if use_cuda else model

# If infersent1 -> use GloVe embeddings. If infersent2 -> use InferSent embeddings.
W2V_PATH = 'save/GloVe/glove.840B.300d.txt' if model_version == 1 else 'save/fastText/crawl-300d-2M.vec'
model.set_w2v_path(W2V_PATH)

# Load embeddings of K most frequent words
model.build_vocab_k_words(K=50000)

# Load some sentences
gen = []
gold = []
with open('InferSent-master/generated_gpt2.txt') as f:
    for line in f:
        gen.append(line)
with open('InferSent-master/gold_gpt2.txt') as f:
    for line in f:
        gold.append(line)

print("gen:", gen[0])
print("gold:", gold[0])

def cosine(u, v):
    return np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))

score = 0
gen_emb = model.encode(gen, bsize=128, tokenize=True, verbose=True)
gold_emb = model.encode(gold, bsize=128, tokenize=True, verbose=True)

for i in range(len(gen)):
    score += cosine(gen_emb[i], gold_emb[i])

print("score:", score/len(gen_emb))

