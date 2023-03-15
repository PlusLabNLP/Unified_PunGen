import torch
import torch.nn.functional as F
import torch.nn as nn
import torchtext

glove = torchtext.vocab.GloVe(name="840B", # trained on Wikipedia 2014 corpus of 6 billion words
                              dim=300)   # embedding size = 100
                              
def get_f(words, pun_word):
    F_scores = []
    for word in words:
        F_scores.append(float(F.cosine_similarity(glove[word], glove[pun_word], dim=0)))
    return F_scores