import sys
from pungen.scorer import LMScorer, SurprisalScorer, UnigramModel, GoodmanScorer
from pungen.options import add_scorer_args, add_generic_args
from pungen.utils import logging_config, get_spacy_nlp
from pungen.wordvec.generate import SkipGram
from pungen.pretrained_wordvec import Glove

import pandas as pd
import numpy as np
import os
import re
import string
import random
import torch
from torch import cuda
from torch import Tensor
import torch.nn as nn
from torch.nn import functional as F
import os
from transformers import GPT2Tokenizer, GPT2LMHeadModel, DistilBertTokenizer, DistilBertForSequenceClassification, BertTokenizer, BertForSequenceClassification

#import amb_dist
from amb_dist import score_examples

#import get_context_and_phrase_new
from get_context_and_phrase_new import generate_input, generate_input_homographic, initialize_wantwords, keywords

#import generate_pun
from generate_pun import generate_pun, generate_pun_gpt3, generate_pun_extract_gpt2, generate_pun_extract_gpt3, generate_pun_vanilla_gpt2, generate_pun_vanilla_gpt3

def get_tabular_format(results, pun_type='homophonic'):
    data = {'pun_sent': [], 'pun_word': [], 'alter_word': [], 'context_word': [], 'phrase': [], 'ambiguity': [], 'distinctiveness_f1': [], 'distinctiveness_f2': []}
    if pun_type == 'homographic':
        data['pun_word_substitute'] = []
        data['alter_word_substitute'] = []
        data.pop('alter_word')
    for each in results:
        data['pun_sent'].append(each['pun_sent'])
        data['pun_word'].append(each['pun_word'])
        if pun_type == 'homophonic':
            data['alter_word'].append(each['alter_word'])
        else:
            data['alter_word_substitute'].append(each['alter_word_substitute'])
            data['pun_word_substitute'].append(each['pun_word_substitute'])
        data['context_word'].append(each['context_word'])
        data['phrase'].append(each['phrase'])
        data['ambiguity'].append(each['scores']['ambiguity'])
        data['distinctiveness_f1'].append(each['scores']['distinctiveness_f1'])
        data['distinctiveness_f2'].append(each['scores']['distinctiveness_f2'])
    data['pun_sent'].append('##Total##')
    data['pun_word'].append('##Total##')
    if pun_type == 'homophonic':
        data['alter_word'].append('##Total##')
    else:
        data['alter_word_substitute'].append('##Total##')
        data['pun_word_substitute'].append('##Total##')
    data['context_word'].append('##Total##')
    data['phrase'].append('##Total##')
    data['ambiguity'].append(sum(data['ambiguity']))
    data['distinctiveness_f1'].append(sum(data['distinctiveness_f1']))
    data['distinctiveness_f2'].append(sum(data['distinctiveness_f2']))
    return pd.DataFrame(data)

def get_a_d_system(gpt2_model_path, label_predictor_path, model_type='gpt2_label_predictor', num_generations_per_pair=5, num_runs=145, output_filepath=os.path.join(os.getcwd(), 'results.csv'), pun_type='homophonic'):    
    
    candidates = []

    gpt2_tokenizer = GPT2Tokenizer.from_pretrained("gpt2-large",    
                            bos_token="<|startoftext|>",
                            eos_token="<|endoftext|>",
                            pad_token="<|pad|>")
    model = GPT2LMHeadModel.from_pretrained(gpt2_model_path).cuda()
    model.resize_token_embeddings(len(gpt2_tokenizer))
    model.eval()
    
    if pun_type == 'homophonic':
        with open('REPO_PATH/overlap_pun_words.txt', 'r') as f1:
            punwords = f1.readlines()
            for i, w in enumerate(punwords):
                punwords[i] = w.strip()
        punwords = punwords[:num_runs]

        with open('REPO_PATH/overlap_alter_words.txt', 'r') as f2:
            alterwords = f2.readlines()
            for i, w in enumerate(alterwords):
                alterwords[i] = w.strip()
        alterwords = alterwords[:num_runs]
                
    elif pun_type == 'homographic':
        with open('REPO_PATH/homographic_punwords.txt', 'r') as f1:
            punwords_homographic = f1.readlines()
            for i, w in enumerate(punwords_homographic):
                punwords_homographic[i] = w.strip()
        punwords_homographic = punwords_homographic[:num_runs]
                
    
    if 'distilbert' in label_predictor_path:
        label_predictor_tokenizer = DistilBertTokenizer.from_pretrained(label_predictor_path)
        label_predictor_model = DistilBertForSequenceClassification.from_pretrained(label_predictor_path).to('cuda')
    else:  
        label_predictor_tokenizer = BertTokenizer.from_pretrained(label_predictor_path)
        label_predictor_model = BertForSequenceClassification.from_pretrained(label_predictor_path).to('cuda')
    label_predictor_model = label_predictor_model.eval()
    
    
    if pun_type == 'homophonic':
        for pun_word, alter_word in zip(punwords, alterwords):
            for i in range(num_generations_per_pair):
                if model_type in['gpt2_label_predictor', 'gpt3_label_predictor', 'gpt2_vanilla', 'gpt3_vanilla']:
                    print("randomly selecting...")
                    contextwords, phrase, _, _ = generate_input(pun_word, alter_word, random_sample=True)
                else:
                    contextwords, phrase, _, _ = generate_input(pun_word, alter_word)
                
                try:
                    select_index = int(len(contextwords)/2)
                    context_word = contextwords[select_index]
                except:
                    continue
                if model_type == 'gpt2_extract':
                    generations = generate_pun_extract_gpt2(model, pun_word, alter_word, phrase, context_word, gpt2_tokenizer, 1)
                elif model_type == 'gpt2_vanilla':
                    generations = generate_pun_vanilla_gpt2(model, pun_word, alter_word, phrase, context_word, gpt2_tokenizer)
                elif model_type == 'gpt2_label_predictor' or model_type == 'gpt2_best':
                    generations = generate_pun(model, pun_word, alter_word, phrase, context_word, label_predictor_model, label_predictor_tokenizer, num_generations_per_pair)
                
                elif model_type in['gpt3_best','gpt3_label_predictor']:
                    generations = generate_pun_gpt3(pun_word, alter_word, phrase, context_word, label_predictor_model, label_predictor_tokenizer, num_generations_per_pair)
                elif model_type =='gpt3_vanilla':
                    generations = generate_pun_vanilla_gpt3(pun_word, alter_word, phrase, context_word, num_generations_per_pair)
                
                elif model_type =='gpt3_extract':
                    generations = generate_pun_extract_gpt3(pun_word, alter_word, phrase, context_word, num_generations_per_pair)
                else:
                    print("Invalid model name")
                    break
                

                for generation in generations:
                    candidates.append({
                        'alter_word': alter_word,
                        'pun_sent': generation,
                        'pun_word': pun_word,
                        'context_word': context_word,
                        'phrase': phrase,
                        'scores': {}
                    })
        results = score_examples(candidates, pun_type='homophonic')
        results_table = get_tabular_format(results, pun_type='homophonic')
        results_table.to_csv(output_filepath, index=False)
        return results_table
    
    elif pun_type == 'homographic':
        for pun_word in punwords_homographic:
            alter_word = pun_word ## for the label predictor, until it's actually trained on homographic puns
            
            try:
                if model_type in['gpt2_label_predictor', 'gpt3_label_predictor', 'gpt2_vanilla', 'gpt3_vanilla']:
                    print("randomly selecting...")
                    contextwords, phrase, _, _, pun_word_def, alt_word_def = generate_input_homographic(pun_word)
                else:
                    contextwords, phrase, _, _, pun_word_def, alt_word_def = generate_input_homographic(pun_word)
            except:
                print('skipping')
                continue
            
            
            context_word = contextwords[0]
            print(pun_word, contextwords, phrase)
            if model_type == 'gpt2_extract':
                generations = generate_pun_vanilla_gpt2(model, pun_word, alter_word, phrase, context_word, gpt2_tokenizer)
            elif model_type == 'gpt2_label_predictor' or model_type == 'gpt2_best':
                generations = generate_pun(model, pun_word, alter_word, phrase, context_word, label_predictor_model, label_predictor_tokenizer)
        
            elif model_type=='gpt3_best':
                generations = generate_pun_gpt3(pun_word, alter_word, phrase, context_word, label_predictor_model, label_predictor_tokenizer, num_generations_per_pair)
            elif model_type == 'gpt3_extract_phrase':
                generations = generate_pun_vanilla_gpt3(pun_word, alter_word, phrase, context_word, num_generations_per_pair)   
                
            ani, tokenizer_En, label_size, word2index_en, index2word_en, index2sememe, index2lexname, index2rootaffix, words_t, wd_sems_, wd_lex, wd_ra, mask_s_, mask_l, mask_r, MODE_en, NUM_RESPONSE = initialize_wantwords()
            
            pun_alter_defs_related_words = keywords([pun_word_def, alt_word_def], ani, tokenizer_En, label_size, word2index_en, index2word_en, index2sememe, index2lexname, index2rootaffix, words_t, wd_sems_, wd_lex, wd_ra, mask_s_, mask_l, mask_r, MODE_en, NUM_RESPONSE)
            
            pun_related_words = list(pun_alter_defs_related_words[0])
            alter_related_words = list(pun_alter_defs_related_words[1])
            pun_related_words.remove(context_word)
            
            for generation in generations:
                candidates.append({
                    'alter_word_substitute': str(alter_related_words[0]),
                    'pun_sent': generation,
                    'pun_word': pun_word,
                    'pun_word_substitute': str(pun_related_words[0]),
                    'context_word': context_word,
                    'phrase': phrase,
                    'scores': {}
                })
        results = score_examples(candidates, pun_type='homographic')
        results_table = get_tabular_format(results, pun_type='homographic')
        results_table.to_csv(output_filepath, index=False)
        return results_table
