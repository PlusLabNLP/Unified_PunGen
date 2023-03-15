from pungen.scorer import LMScorer, SurprisalScorer, UnigramModel, GoodmanScorer
from pungen.options import add_scorer_args, add_generic_args
from pungen.utils import logging_config, get_spacy_nlp
from pungen.wordvec.generate import SkipGram
from pungen.pretrained_wordvec import Glove


import os
import argparse
import json
import numpy as np
import torch
import matplotlib.pyplot as plt
from collections import defaultdict
from scipy.stats import spearmanr, zscore
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import f_regression

import logging
logger = logging.getLogger('pungen')

import nltk
#nltk.download('punkt')
from nltk.tokenize import word_tokenize

args = {
    'lm_path': '/PATH/pungen/models/wikitext/wiki103.pt',
    'word_counts_path': 'REPO_PATH/pungen/models/wikitext/dict.txt',
    'oov_prob': 0.03,
    'skipgram_model': ['REPO_PATH/pungen/data/bookcorpus/skipgram/dict.txt', 'PATH/models/bookcorpus/skipgram/sgns-e15.pt'],
    'skipgram_embed_size': 300,
    # 'human_eval':
    'local_window_size': 2,
    'candidates': [{'alter_word': '',
        'pun_sent': [],
        'pun_word': '',
        'scores': {}
        }],
    'cpu': False
}

def score_examples(candidates, tokenize = True, pun_type='homophonic'):
    unigram_model = UnigramModel(args['word_counts_path'], args['oov_prob'])
    skipgram = SkipGram.load_model(args['skipgram_model'][0], args['skipgram_model'][1], embedding_size=args['skipgram_embed_size'], cpu=args['cpu'])

    glove = None

    scorers = [GoodmanScorer(unigram_model, skipgram, glove)]


    for c in candidates:
        if tokenize:
            c['pun_sent'] = word_tokenize(c['pun_sent'])

        for scorer in scorers:
            if pun_type == 'homophonic':
                scores = scorer.analyze(c['pun_sent'], c['pun_word'], c['alter_word'])
            else:
                scores = scorer.analyze(c['pun_sent'], c['pun_word_substitute'], c['alter_word_substitute'])
            c['scores'].update(scores)

    return candidates

### FUNCTION TO GET A, D FOR A SENTENCE -> Input: sentence -> string, pun_word -> string, alter_word -> string, args = args
def get_a_d_for_sentence(sentence, pun_word, alter_word, args=args):
    tokens = word_tokenize(sentence)
    candidate = [{'pun_sent': tokens, 'pun_word': pun_word, 'alter_word': alter_word, 'scores': {}},]
    results = score_examples(candidate, tokenize = False)
    a = results[0]['scores']['ambiguity']
    d_f1 = results[0]['scores']['distinctiveness_f1']
    d_f2 = results[0]['scores']['distinctiveness_f2']
    return torch.tensor(a).reshape([1]), torch.tensor(d_f1).reshape([1]), torch.tensor(d_f2).reshape([1])

if __name__ == '__main__':
    pass
