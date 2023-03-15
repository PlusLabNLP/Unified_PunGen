import torch
import torch.nn.functional as F
import torch.nn as nn
import torchtext
from transformers import GPT2Tokenizer

glove = torchtext.vocab.GloVe(name="840B", # trained on Wikipedia 2014 corpus of 6 billion words
                              dim=300)   # embedding size = 100

def score_examples_f1f2(candidates, next_words=None):
    if next_words == None:
        tokenizer = GPT2Tokenizer.from_pretrained('gpt2-large')
        next_words = []
        for i in range(len(tokenizer)):
            word_here = tokenizer.convert_ids_to_tokens(i)
            next_words.append(word_here)
        
    for c in candidates:    
        scores_f1f2 = {}
        for next_word in next_words:
            if next_word[0] == 'Ġ':
                next_word_without_g = next_word[1:]
            else:
                next_word_without_g = next_word
                
            if all(glove[next_word_without_g] == torch.tensor([0] * 300)):
                score_f1 = -1
                score_f2 = -1
            else:
                glove_punword = glove[c['pun_word']]
                glove_alterword = glove[c['alter_word']]
                score_f1 = F.cosine_similarity(glove[next_word_without_g], glove_punword, dim=0)
                score_f2 = F.cosine_similarity(glove[next_word_without_g], glove_alterword, dim=0)
            scores_f1f2[next_word] = {'f1_score': score_f1, 'f2_score': score_f2}
        c['scores_f1f2'] = scores_f1f2
    
    return candidates