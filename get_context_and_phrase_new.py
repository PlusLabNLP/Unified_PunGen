import numpy as np
import pandas as pd
import json
import os
import re
import torch
import transformers
import random
from random import randrange
import requests
from pprint import pprint
import string
from torch import cuda
import os
from transformers import pipeline
import nltk
from nltk.corpus import webtext
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from nltk.probability import FreqDist
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet as wn
import pickle as pkl

from transformers import AutoModelWithLMHead, AutoTokenizer, BertTokenizer
import torch
from transformers import pipeline
from transformers import FillMaskPipeline
rank_phrases_tokenizer = AutoTokenizer.from_pretrained("roberta-large")
rank_phrases_model = AutoModelWithLMHead.from_pretrained("roberta-large")
rank_phrases_pipeline = FillMaskPipeline(model=rank_phrases_model, tokenizer=rank_phrases_tokenizer, task='fill-mask')


import torch
import torch.nn as nn
import torch.optim as optim
import torch, gc, json, os, string, re, requests, hashlib, urllib.parse 
import numpy as np
from datetime import datetime
from pytorch_transformers import *
import sys
sys.path.append("PATH/WantWords")

with open('PATH/wiki_guten_colbert.txt', 'r') as f1:
    wordfrequency_corpus = f1.readlines()

with open('PATH/wordfrequency_corpus.pkl', 'rb') as f2:
    words = pkl.load(f2)
    
data_analysis = nltk.FreqDist(words)
synsets_txt_filepath = 'PATH/synsets.txt'

### WSD - EWISER
import spacy
from ewiser.spacy.disambiguate import Disambiguator
path_to_english_torch_model_checkpoint = "PATH/WSD_Model/pytorch_model.pt"
nlp = spacy.load("en_core_web_sm", disable=['parser', 'ner'])
wsd = Disambiguator(path_to_english_torch_model_checkpoint, lang="en")
wsd.enable(nlp, "wsd")


def get_predictions_ewiser(sentence):
    results = {}
    doc = nlp(sentence)
    for w in doc:
        if w._.offset:
            # print(w.text, w.lemma_, w.pos_, w._.offset, w._.synset.definition())
            results[w.text] = w._.synset.definition()
    return results
### WSD - EWISER


### STOPWORDS
stopwords_en = nlp.Defaults.stop_words
### STOPWORDS


def check_if_any_uppercase_or_punctuation(test_str):
    punct = string.punctuation
    if any(letter.isupper() for letter in test_str):
        return True
    elif any(char in punct for char in test_str):
        return True
    else:
        return False
        
def check_stopwords_count(word, line):
    tokenized_line = line.split()
    for i, word in enumerate(tokenized_line):
        if word[-1] in string.punctuation:
            tokenized_line[i] = word[:-1]
        elif word[0] in string.punctuation:
            tokenized_line[i] = word[1:]
    stopwords_in_line = [word for word in tokenized_line if word in stopwords_en]
    return len(stopwords_in_line)
    
def get_word_definition(punword, word_with_surrounding_context):
     #url = "https://api.dictionaryapi.dev/api/v2/entries/en/" + word
     #result = requests.get(url=url).json()
     #word_definition = result[0]['meanings'][0]['definitions'][0]['definition']
     #return word_definition
    result = get_predictions_ewiser(word_with_surrounding_context)
    if punword not in result.keys():
        print("# EWISER definition for the word '" + punword + "' in the context '" + word_with_surrounding_context + "' not found. Using definition from api.dictionaryapi.dev. This should NOT happen if this is being used for homographic puns.")
        url = "https://api.dictionaryapi.dev/api/v2/entries/en/" + punword
        result = requests.get(url=url).json()
        word_definition = result[0]['meanings'][0]['definitions'][0]['definition']
        print("Definition obtained: " + word_definition)
        return word_definition
    else:
        return result[punword]

## WANTWORDS - related words from sense def
def initialize_wantwords(wantwords_base_path='PATH/BART-Approach/WantWords/', device=torch.device('cpu')):

    ani= torch.load(wantwords_base_path + "website_RD/1rsl_saved.model", map_location=lambda storage, loc: storage)
    tokenizer_class = BertTokenizer
    tokenizer_Ch = tokenizer_class.from_pretrained('bert-base-chinese')
    tokenizer_En = tokenizer_class.from_pretrained('bert-base-uncased')
    (_, (_, label_size, _, _), (word2index_en, index2word_en, index2sememe, index2lexname, index2rootaffix)) = np.load(wantwords_base_path + "website_RD/"+ 'data_inUse1_en.npy', allow_pickle=True)
    device = torch.device('cpu')
    words_t = torch.tensor(np.array([0]))
    
    (data_train_idx, data_dev_idx, data_test_500_seen_idx, data_test_500_unseen_idx, data_defi_c_idx, data_desc_c_idx) = np.load(wantwords_base_path + "website_RD/" + 'data_inUse2_en.npy', allow_pickle=True)
    data_all_idx = data_train_idx + data_dev_idx + data_test_500_seen_idx + data_test_500_unseen_idx + data_defi_c_idx
    index2word_en = np.array(index2word_en)
    #print('-------------------------2')
    sememe_num = len(index2sememe)
    wd2sem = word2feature(data_all_idx, label_size, sememe_num, 'sememes')
    wd_sems_ = label_multihot(wd2sem, sememe_num)
    wd_sems_ = torch.from_numpy(np.array(wd_sems_)).to(device) 
    lexname_num = len(index2lexname)
    wd2lex = word2feature(data_all_idx, label_size, lexname_num, 'lexnames') 
    wd_lex = label_multihot(wd2lex, lexname_num)
    wd_lex = torch.from_numpy(np.array(wd_lex)).to(device)
    rootaffix_num = len(index2rootaffix)
    wd2ra = word2feature(data_all_idx, label_size, rootaffix_num, 'root_affix') 
    wd_ra = label_multihot(wd2ra, rootaffix_num)
    wd_ra = torch.from_numpy(np.array(wd_ra)).to(device)
    mask_s_ = mask_noFeature(label_size, wd2sem, sememe_num)
    mask_l = mask_noFeature(label_size, wd2lex, lexname_num)
    mask_r = mask_noFeature(label_size, wd2ra, rootaffix_num)

    MODE_en = 'rsl'
    itemsPerCol = 20
    GET_NUM = 100
    NUM_RESPONSE = 500
    
    return ani, tokenizer_En, label_size, word2index_en, index2word_en, index2sememe, index2lexname, index2rootaffix, words_t, wd_sems_, wd_lex, wd_ra, mask_s_, mask_l, mask_r, MODE_en, NUM_RESPONSE

def Score2Hexstr(score, maxsc):
    thr = maxsc/1.5
    l = len(score)
    ret = ['00']*l
    for i in range(l):
        res = int(200*(score[i] - thr)/thr)
        if res>15:
            ret[i] = hex(res)[2:]
        else:
            break
    return ret

def label_multihot(labels, num):
    sm = np.zeros((len(labels), num), dtype=np.float32)
    for i in range(len(labels)):
        for s in labels[i]:
            if s >= num:
                break
            sm[i, s] = 1
    return sm

def word2feature(dataset, word_num, feature_num, feature_name, device=torch.device('cpu')):
    max_feature_num = max([len(instance[feature_name]) for instance in dataset])
    ret = np.zeros((word_num, max_feature_num), dtype=np.int64)
    ret.fill(feature_num)
    for instance in dataset:
        if ret[instance['word'], 0] != feature_num: 
            continue # this target_words has been given a feature mapping, because same word with different definition in dataset
        feature = instance[feature_name]
        ret[instance['word'], :len(feature)] = np.array(feature)
    return torch.tensor(ret, dtype=torch.int64, device=device)

def mask_noFeature(label_size, wd2fea, feature_num, device=torch.device('cpu')):
    mask_nofea = torch.zeros(label_size, dtype=torch.float32, device=device)
    for i in range(label_size):
        feas = set(wd2fea[i].detach().cpu().numpy().tolist())-set([feature_num])
        if len(feas)==0:
            mask_nofea[i] = 1
    return mask_nofea

def keywords(sentences, ani, tokenizer_En, label_size, word2index_en, index2word_en, index2sememe, index2lexname, index2rootaffix, words_t, wd_sems_, wd_lex, wd_ra, mask_s_, mask_l, mask_r, MODE_en, NUM_RESPONSE, device=torch.device('cpu')):
  output=[]
  for description in sentences:
  
    RD_mode='EE'
    with torch.no_grad():
      def_words = re.sub('[%s]' % re.escape(string.punctuation), ' ', description)
      def_words =def_words.lower()
      def_words = def_words.strip().split()
      def_word_idx = []
      # print("def words:", def_words)
      if len(def_words) > 0:
        for def_word in def_words:
            if def_word in word2index_en:
              def_word_idx.append(word2index_en[def_word])
            else:
              # print("hi")
              def_word_idx.append(word2index_en['<OOV>'])
        # print("def_word_idx: ", def_word_idx, print(word2index_en['<OOV>']),set(def_word_idx),{word2index_en['<OOV>']})
        x_len = len(def_word_idx)
        if set(def_word_idx)=={word2index_en['<OOV>']}:
            x_len = 1
        if x_len==1:
            if def_word_idx[0]>1:
                
                score = ((model_en.embedding.weight.data).mm((model_en.embedding.weight.data[def_word_idx[0]]).unsqueeze(1))).squeeze(1)
                if RD_mode=='EE': 
                    score[def_word_idx[0]] = -10.
                score[np.array(index2synset_en[def_word_idx[0]])] *= 2
                sc, indices = torch.sort(score, descending=True)
                predicted = indices[:NUM_RESPONSE].detach().cpu().numpy()

                score = sc[:NUM_RESPONSE].detach().numpy()
                maxsc = sc[0].detach().item()
                s2h = Score2Hexstr(score, maxsc)
            else:
                predicted= []
                ret = {'error': 1} 
        else:
            defi = '[CLS] ' + description
            def_word_idx = tokenizer_En.encode(defi)[:60]
            def_word_idx.extend(tokenizer_En.encode('[SEP]'))
            definition_words_t = torch.tensor(np.array(def_word_idx), dtype=torch.int64, device=device)
            definition_words_t = definition_words_t.unsqueeze(0) # batch_size = 1
            score = ani('test', x=definition_words_t, w=words_t, ws=wd_sems_, wl=wd_lex, wr=wd_ra, msk_s=mask_s_, msk_l=mask_l, msk_r=mask_r, mode=MODE_en)
            sc, indices = torch.sort(score, descending=True)
            predicted = indices[0, :NUM_RESPONSE].detach().cpu().numpy()
            score = sc[0, :NUM_RESPONSE].detach().numpy()
            maxsc = sc[0, 0].detach().item()
            s2h = Score2Hexstr(score, maxsc)
              
      else:
        print("hiiii")
        predicted= []
        ret = {'error': 0} 
    # print("predicted: ", predicted)
    if len(predicted)>0:
        res = index2word_en[predicted]
        # print("res: ",res)
        ret = [] 
        cn = -1
        if RD_mode == "EE":
            def_words = set(def_words)
            for wd in res:
                cn += 1
                if len(wd)>1 and (wd not in def_words):
                    # ret.append(wd_data_en[wd]) # wd_data_en[wd] = {'word': word, 'definition':defis, 'POS':['n']}]
                    # ret[len(ret)-1]['c'] = s2h[cn]
                    try:
                        ret.append(wd_data_en[wd]) # wd_data_en[wd] = {'word': word, 'definition':defis, 'POS':['n']}]
                        ret[len(ret)-1]['c'] = s2h[cn]
                    except:
                        continue
        else:
            for wd in res:
                cn += 1
                if len(wd)>1:
                    try:
                        ret.append(wd_data_en[wd]) # wd_data_en[wd] = {'word': word, 'definition':defis, 'POS':['n']}]
                        ret[len(ret)-1]['c'] = s2h[cn]
                    except:
                        continue
    output.append(res[0:8])
  return output

def score_phrase(phrase_with_mask, target_word):
    if target_word[0] == " ":
        target_word_g = "Ġ" + target_word[1:]
    else:
        target_word_g = target_word
    if target_word_g in rank_phrases_tokenizer.vocab:
        scored_phrase = rank_phrases_pipeline.__call__(phrase_with_mask, targets=target_word_g)
        return scored_phrase
    else:
        scores = []
        mask_insert_position = phrase_with_mask.find(rank_phrases_tokenizer.mask_token)
        tokenized_target_word_g = rank_phrases_tokenizer.tokenize(target_word)    
        while len(tokenized_target_word_g) > 0:
            token_to_insert = tokenized_target_word_g[0]
            tokenized_target_word_g.pop(0)
            scored_phrase = rank_phrases_pipeline.__call__(phrase_with_mask, targets=token_to_insert)
            scores.append(scored_phrase[0]['score'])
            if token_to_insert[0] == 'Ġ':
                mask_insert_position += len(token_to_insert) - 1
            else:
                mask_insert_position += len(token_to_insert)
            phrase_with_mask = str(scored_phrase[0]['sequence'][:mask_insert_position]) + str(rank_phrases_tokenizer.mask_token) + str(scored_phrase[0]['sequence'][mask_insert_position:])
#         print(scores)
        product_of_scores = 1
        for each in scores:
            product_of_scores *= each
        return [{'score': product_of_scores, 'sequence': scored_phrase[0]['sequence'], 'token': scored_phrase[0]['token'], 'token_str': target_word}]

def rank_candidate_phrases(phrases, pun_word, alt_word, word_to_rank='pun'):
    scored_phrases_dict = {}
    for phrase in phrases:
        try:
            phrase_with_mask = phrase.replace(alt_word, rank_phrases_tokenizer.mask_token, 1)
            target_word = (pun_word if word_to_rank == 'pun' else alt_word)
            scored_phrase = score_phrase(phrase_with_mask, target_word=" " + target_word)
            scored_phrases_dict[scored_phrase[0]['sequence']] = scored_phrase[0]['score']
        except:
            continue
    return scored_phrases_dict

def get_pun_contextword(word_with_surrounding_context, alt_word, pun_word, data_analysis, punword_definition, pun_type='homophonic', random_sample = False):
    if pun_type == 'homophonic':
        word_with_surrounding_context = word_with_surrounding_context.replace(alt_word+' ', pun_word+' ')
    
    if pun_type == 'homophonic' and punword_definition == None:
        punword_definition = get_word_definition(pun_word, word_with_surrounding_context)
        print("Definition obtained for the word '" + pun_word + "': " + punword_definition)
              
    ani, tokenizer_En, label_size, word2index_en, index2word_en, index2sememe, index2lexname, index2rootaffix, words_t, wd_sems_, wd_lex, wd_ra, mask_s_, mask_l, mask_r, MODE_en, NUM_RESPONSE = initialize_wantwords()
    
    wantwords = keywords([punword_definition], ani, tokenizer_En, label_size, word2index_en, index2word_en, index2sememe, index2lexname, index2rootaffix, words_t, wd_sems_, wd_lex, wd_ra, mask_s_, mask_l, mask_r, MODE_en, NUM_RESPONSE)
    
    context_words = [str(word) for word in wantwords[0]]

    if random_sample == True:
        context_words = random.sample(context_words, 1)
    context_words_and_frequency_scores = {}
    for word in context_words:
        if pun_word not in word and word not in pun_word:
            context_words_and_frequency_scores[word] = 0
    
    for word in context_words_and_frequency_scores.keys():
        context_words_and_frequency_scores[word] = data_analysis[word]
    
    return word_with_surrounding_context, context_words_and_frequency_scores

def get_maxfreq_contextword(context_words_and_frequency_scores, N = 1):
    x = context_words_and_frequency_scores
    toreturn = [k for k, v in sorted(x.items(), key=lambda item: item[1], reverse=True)]
    return toreturn[:N]

def extract_surrounding_context(word, lines, pun_word, N = 20, index = 1, random_sample = False):
    contexts_containing_the_word = []
    for line in lines:
        tokenized_line = line.split()
        if word in tokenized_line:
            word_index = tokenized_line.index(word)
            if word_index >= 3 and word_index < len(tokenized_line) - 4:
                extraction_lower_bound = word_index - 3
                extraction_upper_bound = word_index + 4
                context_around_word = tokenized_line[extraction_lower_bound:extraction_upper_bound]
                context_around_word_joined = " ".join(context_around_word)
                stopwords_count = check_stopwords_count(word, context_around_word_joined)
                if stopwords_count <= len(context_around_word) - 3 and check_if_any_uppercase_or_punctuation(context_around_word_joined) == False:
                    contexts_containing_the_word.append(context_around_word_joined)
    
    if len(contexts_containing_the_word) == 0:
        raise ValueError("Alternative word not found in the corpus")
    else:
    
        if random_sample==True:
            return random.sample(contexts_containing_the_word, 1)[0], []
        elif len(contexts_containing_the_word) >= N:
            contexts_containing_the_word = random.sample(contexts_containing_the_word, N)

#         random_context = random.choice(contexts_containing_the_word)
#         for i, context in enumerate(contexts_containing_the_word):
#             contexts_containing_the_word[i] = context.replace(word, pun_word)
        altword_ranked_phrases = rank_candidate_phrases(contexts_containing_the_word, pun_word, word, word_to_rank='alt')
        altword_ranked_phrases_1 = sorted(altword_ranked_phrases, key=altword_ranked_phrases.get, reverse=True)[:8]

        ranked_phrases = rank_candidate_phrases(altword_ranked_phrases_1, pun_word, word, word_to_rank='pun')
        ranked_phrases_1 = sorted(ranked_phrases, key=ranked_phrases.get, reverse=True)

        index = int(len(ranked_phrases_1)/2)
        ranked_bestphrase = ranked_phrases_1[index]
        if not pun_word in ranked_phrases_1[0]:
            print("Phrase ranking doesn't work- the tokenizer doesn't have the complete pun word token. Randomly selecting.")
            ranked_bestphrase = random.choice(contexts_containing_the_word)

        return ranked_bestphrase, ranked_phrases

def extract_surrounding_context_homographic(pun_word, alt_word_def, lines, N, select_phrase_at_index):
    contexts_containing_the_word = []
    for line in lines:
        if len(contexts_containing_the_word) < N:
            tokenized_line = line.split()
            if pun_word in tokenized_line:
                try:
                    wsd_prediction = get_predictions_ewiser(line)[pun_word]
                except:
                    continue
                if wsd_prediction == alt_word_def:
                    word_index = tokenized_line.index(pun_word)
                    if word_index >= 3 and word_index < len(tokenized_line) - 4:
                        extraction_lower_bound = word_index - 3
                        extraction_upper_bound = word_index + 4
                        context_around_word = tokenized_line[extraction_lower_bound:extraction_upper_bound]
                        context_around_word_joined = " ".join(context_around_word)
                        stopwords_count = check_stopwords_count(pun_word, context_around_word_joined)
                        if stopwords_count <= len(context_around_word) - 3 and check_if_any_uppercase_or_punctuation(context_around_word_joined) == False:
                            contexts_containing_the_word.append(context_around_word_joined)
    if len(contexts_containing_the_word) == 0:
        raise ValueError("Alternative definition not found in the corpus")
    else:
        phrase_at_index = contexts_containing_the_word[select_phrase_at_index]
        return phrase_at_index, contexts_containing_the_word


def generate_input(pun_word, alt_word, pun_def = None, N = 2, index = 0, random_sample = False):
    ranked_bestphrase, ranked_phrases = extract_surrounding_context(word = alt_word, lines = wordfrequency_corpus, pun_word=pun_word, index=index, random_sample=random_sample)
    word_with_surrounding_context, context_words_and_frequency_scores = get_pun_contextword(ranked_bestphrase, alt_word, pun_word, data_analysis, pun_def,  pun_type='homophonic',random_sample=random_sample)
    context_word = get_maxfreq_contextword(context_words_and_frequency_scores, N = 5)
    return context_word, word_with_surrounding_context, context_words_and_frequency_scores, ranked_phrases    


# def synset_from_sense_key(sense_key):
#     sense_key_regex = r"(.*)\%(.*):(.*):(.*):(.*):(.*)"
#     synset_types = {1:'n', 2:'v', 3:'a', 4:'r', 5:'s'}
#     lemma, ss_type, lex_num, lex_id, head_word, head_id = re.match(sense_key_regex, sense_key).groups()
#     ss_idx = '.'.join([lemma, synset_types[int(ss_type)], lex_id])
#     return wordnet.synset(ss_idx)

def synset_from_sense_key(sense_key):
    keys = [sense_key]
    new_keys = []
    for i, key in enumerate(keys):
        if ';' in key:
            keys[i] = key.split(';')[0]
    defs = []
    for key in keys:
        synset_key = str(wn.lemma_from_key(key)).split("'")[1]
        sk_split = synset_key.split('.')
        sk = ''
        for part in sk_split[:-1]:
            sk += part + '.'
        sk = sk[:-1]
        # print(sk)
        def1 = wn.synset(sk)#.definition()
        defs.append(def1)
    return defs[0]

def get_sensedefs_homographic(pun_word, synsets_txt_filepath, lemmatizer=WordNetLemmatizer()):
    if len(pun_word.split()) > 1:
        pun_word = pun_word.replace(' ', '_')
    with open(synsets_txt_filepath, 'r') as f1:
        lines = f1.readlines()
    def_1 = None
    def_2 = None
    for line in lines:
        defs = line.split()
        sensekey_1 = defs[0].split(';')[0]
        sensekey_2 = defs[1].split(';')[0]
        word_1 = sensekey_1.split('%')[0]
        word_2 = sensekey_2.split('%')[0]
        if pun_word == word_1 and pun_word == word_2:
            print("Wordnet sense key 1: ", sensekey_1)
            print("Wordnet sense key 2: ", sensekey_2)
            try:
                def_1 = synset_from_sense_key(sensekey_1).definition()
                def_2 = synset_from_sense_key(sensekey_2).definition()
                break
            except:
                print("Synset not found for sense keys '" + str(sensekey_1) + ", " + str(sensekey_2))
                continue
    if def_1 != None and def_2 != None:
        return def_1, def_2
    else:
        pun_word = lemmatizer.lemmatize(pun_word)
        def_1 = None
        def_2 = None
        for line in lines:
            defs = line.split()
            sensekey_1 = defs[0].split(';')[0]
            sensekey_2 = defs[1].split(';')[0]
            word_1 = sensekey_1.split('%')[0]
            word_2 = sensekey_2.split('%')[0]
            if pun_word == word_1 and pun_word == word_2:
                print("Wordnet sense key 1: ", sensekey_1)
                print("Wordnet sense key 2: ", sensekey_2)
                try:
                    def_1 = synset_from_sense_key(sensekey_1).definition()
                    def_2 = synset_from_sense_key(sensekey_2).definition()
                    break
                except:
                    print("Synset not found for sense keys " + str(sensekey_1) + ", " + str(sensekey_2))
                    continue
    if def_1 != None and def_2 != None:
        return def_1, def_2
    else:
        raise Exception('Pun word not found in the SemEval sensedef data')


def generate_input_homographic(pun_word, pun_word_def=None, alt_word_def=None, N=2, select_phrase_at_index=0, synsets_txt_filepath="PATH/synsets.txt"):
    if pun_word_def == None or alt_word_def == None:
        pun_word_def, alt_word_def = get_sensedefs_homographic(pun_word, synsets_txt_filepath)
    #print(alt_word_def)
    phrase_at_index, contexts_containing_the_word = extract_surrounding_context_homographic(pun_word, alt_word_def, wordfrequency_corpus, N, select_phrase_at_index)
#     print("phrase at index = ", phrase_at_index)
#     print("contexts containing the word = ", contexts_containing_the_word)
    word_with_surrounding_context, context_words_and_frequency_scores = get_pun_contextword(phrase_at_index, pun_word, pun_word, data_analysis, pun_word_def, pun_type='homographic')
#     print("word with surrounding context = ", word_with_surrounding_context)
#     print("context words and frequency scores = ", context_words_and_frequency_scores)
    context_word = get_maxfreq_contextword(context_words_and_frequency_scores, N = N)
#     print("context word = ", context_word)
    return context_word, word_with_surrounding_context, context_words_and_frequency_scores, contexts_containing_the_word, pun_word_def, alt_word_def

