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
MODEL_PATH = "encoder/infersent%s.pkl" % model_version
params_model = {'bsize': 64, 'word_emb_dim': 300, 'enc_lstm_dim': 2048,
                'pool_type': 'max', 'dpout_model': 0.0, 'version': model_version}
model = InferSent(params_model)
model.load_state_dict(torch.load(MODEL_PATH))

# Keep it on CPU or put it on GPU
use_cuda = False
model = model.cuda() if use_cuda else model

# If infersent1 -> use GloVe embeddings. If infersent2 -> use InferSent embeddings.
W2V_PATH = 'GloVe/glove.840B.300d.txt' if model_version == 1 else 'fastText/crawl-300d-2M.vec'
model.set_w2v_path(W2V_PATH)

# Load embeddings of K most frequent words
model.build_vocab_k_words(K=12)

# Load some sentences
sentences = []
gen = []
gold = []
with open('InferSent-master/generated_gpt2.txt') as f:
    for line in f:
        sentences.append(line.strip())
        gen.append(line)
with open('InferSent-master/gold_gpt2.txt') as f:
    for line in f:
        sentences.append(line.strip())
        gold.append(line)

print(len(sentences))
print("gen:", gen[0])
print("gold:", gold[0])

embeddings = model.encode(sentences, bsize=128, tokenize=False, verbose=True)
print('nb sentences encoded : {0}'.format(len(embeddings)))

print(np.linalg.norm(model.encode(gen[0])))

def cosine(u, v):
    return np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))

print(cosine(model.encode(['                 '])[0], model.encode('a valid sentence')[0]))
score = 0
for i in range(len(gen)):
    score += cosine(model.encode(gen[i])[0], model.encode(gold[i])[0])

print("score:", score/len(gen))

