import sys
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
import openai
from transformers import GPT2Tokenizer, GPT2LMHeadModel, DistilBertTokenizer, DistilBertForSequenceClassification, BertTokenizer, BertForSequenceClassification

from get_F_score import get_f

tokenizer = GPT2Tokenizer.from_pretrained("gpt2-large",    
                        bos_token="<|startoftext|>",
                        eos_token="<|endoftext|>",
                        pad_token="<|pad|>")
openai.api_key = APIKEY
# def get_freer_gpu():
#     os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >tmp')
#     memory_available = [int(x.split()[2]) for x in open('tmp', 'r').readlines()]
#     return np.argmax(memory_available)
device = 'cuda'
def check_phrase(last_word, phrase):
    if last_word in phrase.split():
        i = phrase.split().index(last_word)
        if i < len(phrase.split())-1:
            return True, phrase.split()[i+1]
    return False, ''
def clean(input_ids1_l, results, punc='.', treshold = 3):
    boolean_list = [punc in res for res in results]
    if sum(boolean_list)>=treshold:
        indices_to_keep = [i for i,x in enumerate(boolean_list) if x == True]
        return [input_ids1_l[i] for i in indices_to_keep], [results[i] for i in indices_to_keep]
    else:
        return input_ids1_l, results
    
def predict_nextlabel_given_model(sent, pun_word, alter_word, label_predictor_model, label_predictor_tokenizer, device='cuda'):

    softmax = nn.Softmax(dim=1)
    #label_dict = {0: 'A', 1: 'D_F1', 2: 'D_F2'}

    input_sent = 'Pun Sentence: ' + sent + ' [SEP] Pun word: ' + pun_word + ' [SEP] Alternative Word: ' + alter_word
    input_ids = torch.tensor([label_predictor_tokenizer(input_sent)['input_ids']]).to(device)
    attention_mask = torch.tensor([label_predictor_tokenizer(input_sent)['attention_mask']]).to(device)
    outputs = label_predictor_model(input_ids=input_ids, attention_mask=attention_mask)
    probs = softmax(outputs.logits).detach().to('cpu')[0]
    label = int(torch.argmax(probs).detach().to('cpu'))
    return probs.tolist(), label

def top_k_top_p_filtering(
    logits: Tensor,
    top_k: int = 0,
    top_p: float = 1.0,
    filter_value: float = -float("Inf"),
    min_tokens_to_keep: int = 1,
    return_index = False
) -> Tensor:
    """Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
    Args:
        logits: logits distribution shape (batch size, vocabulary size)
        if top_k > 0: keep only top k tokens with highest probability (top-k filtering).
        if top_p < 1.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
            Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
        Make sure we keep at least min_tokens_to_keep per batch example in the output
    From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
    """
    if top_k > 0:
        top_k = min(max(top_k, min_tokens_to_keep), logits.size(-1))  # Safety check
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        indices_keep = logits >= torch.topk(logits, top_k)[0][..., -1, None]
        indices_keep = indices_keep[0].tolist()
        indices_keep = [i for i,x in enumerate(indices_keep) if x == True]
        logits[indices_to_remove] = filter_value

    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold (token with 0 are kept)
        sorted_indices_to_remove = cumulative_probs > top_p
        if min_tokens_to_keep > 1:
            # Keep at least min_tokens_to_keep (set to min_tokens_to_keep-1 because we add the first one below)
            sorted_indices_to_remove[..., :min_tokens_to_keep] = 0
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        # scatter sorted tensors to original indexing
        indices_to_remove = sorted_indices_to_remove.scatter(-1, sorted_indices, sorted_indices_to_remove)
        logits[indices_to_remove] = filter_value
    if return_index == True:
        return logits, indices_keep
    return logits

def generate_next_word(model, tokenizer, input_ids1, temperature = 0.85):
    current_word = 0
    for _ in range(10):
        outputs1 = model(input_ids1)
        next_token_logits1 = outputs1[0][:, -1, :]
        next_token_logits1 = top_k_top_p_filtering(next_token_logits1, top_k=100)
        logit_zeros = torch.zeros(len(next_token_logits1), device=device)

        next_token_logits = next_token_logits1 * temperature
        probs = F.softmax(next_token_logits, dim=-1)
        next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
        unfinished_sents = torch.ones(1, dtype=torch.long, device=device)
        tokens_to_add = next_tokens * unfinished_sents + tokenizer.pad_token_id * (1 - unfinished_sents)

        if tokenizer.eos_token_id in next_tokens[0]:
            input_ids1 = torch.cat([input_ids1, tokens_to_add.unsqueeze(-1)], dim=-1)
            return input_ids1, True

        #print(tokenizer.decode(input_ids1[0]))
        if tokenizer.decode(tokens_to_add[0])[0] == ' ':
            if current_word ==1:
                return input_ids1, False
            current_word += 1
        input_ids1 = torch.cat([input_ids1, tokens_to_add.unsqueeze(-1)], dim=-1)
    raise Error

def generate_next_word_new(model, input_ids1, top_k, temperature=1, num_samples=5):
    original = tokenizer.decode(input_ids1[0])
    for _ in range(2):
        outputs1 = model(input_ids1)
        next_token_logits1 = outputs1[0][:, -1, :]
        next_token_logits1 = top_k_top_p_filtering(next_token_logits1, top_k=top_k)
        logit_zeros = torch.zeros(len(next_token_logits1), device=device)

        next_token_logits = next_token_logits1 / temperature
        probs = F.softmax(next_token_logits, dim=-1)
        next_tokens = torch.multinomial(probs, num_samples=num_samples).squeeze(1)
        unfinished_sents = torch.ones(1, dtype=torch.long, device=device)
        tokens_to_add = next_tokens * unfinished_sents + tokenizer.pad_token_id * (1 - unfinished_sents)

        temp = []
        for i in range(len(input_ids1)):
            temp +=[torch.cat([input_ids1[i].reshape(1,-1), token_to_add.reshape(1,-1)], dim=-1) for token_to_add in tokens_to_add[i]]
        input_ids1 = torch.stack(temp).view(len(temp),-1)
        
    results = []
    input_ids1_l = []
    for input_id1 in input_ids1:
        gen = tokenizer.decode(input_id1).replace(original,'').strip(' ')
        if len(gen.split()) >0:
            gen = gen.split()[0]
            if gen not in results:
                results.append(gen)
                input_ids1_l.append(tokenizer(original+' '+gen+' ',   return_tensors='pt').input_ids.cuda())
        
        
    return input_ids1_l, results, '<|endoftext|>' in list(set(results))
    
def generate_next_word_gpt3(input_text, T=0.9, num_samples=5):
    response = openai.Completion.create(
  engine="davinci-instruct-beta",
  prompt=input_text,
  temperature=T,
  max_tokens=5,
  top_p=0.95,
  n = num_samples)
    temp = [r['text'].strip('\n').split() for r in response['choices']]
    candidates, input_text_l,finish_reason, eos = [], [],[], False
    
    for r in response['choices']:
        if 'pun' in r['text'] or "Pun" in r['text']:
            eos = True
    
    if not eos:
        for t in temp:
            if len(t)>0 and t[0] not in candidates:
                candidates.append(t[0])
                input_text_l.append(input_text+' '+t[0])
        
    
        
    return input_text_l, candidates,  eos or len(candidates)==0

label_dict = {0: 'A', 1: 'D_F1', 2: 'D_F2'}

def generate_pun_extract_gpt2(model, pun_word, alter_word, phrase, context_word, gpt2_tokenizer, num_generations=5):
    prompt1 = 'Generate a sentence with the context word "' + context_word + '" and the phrase "' + phrase + '":' 
    input_ids1 = gpt2_tokenizer(prompt1,   
                          return_tensors='pt').input_ids.cuda()
    generations = []
    for i in range(num_generations):
        ans = model.generate(input_ids1, do_sample=True, 
                         top_k=30, 
                         max_length=80,  
                         # Changes randomness of generated sequences
                         temperature=0.9,
                         # Number of sequences to generate                 
                         num_return_sequences=1)
        # Print generated descriptions
        for a in ans:
            a = gpt2_tokenizer.decode(a, skip_special_tokens=True).replace(prompt1,'')
            generations.append(a)
    
    return generations

def generate_pun_vanilla_gpt2(model, pun_word, alter_word, phrase, context_word, gpt2_tokenizer, num_generations=5):
    prompt1 = f'Pun word: {pun_word}, Alter word: {alter_word}, Generate a sentence :' 
    input_ids1 = gpt2_tokenizer(prompt1,   
                          return_tensors='pt').input_ids.cuda()
    generations = []
    for i in range(num_generations):
        ans = model.generate(input_ids1, do_sample=True, 
                         top_k=30, 
                         max_length=80,  
                         # Changes randomness of generated sequences
                         temperature=0.9,
                         # Number of sequences to generate                 
                         num_return_sequences=1)
        # Print generated descriptions
        for a in ans:
            a = gpt2_tokenizer.decode(a, skip_special_tokens=True).replace(prompt1,'')
            generations.append(a)
    
    return generations



def generate_pun(model, pun_word, alter_word, phrase, context_word, label_predictor_model, label_predictor_tokenizer):
    max_len = 35
    T = 0.9
    top_k = 30
    i = 0
    generated_puns = []

    prompt1 = 'Generate a sentence with the context word "' + context_word + '" and the phrase "' + phrase + '": '
    while len(generated_puns) <1 and i<10:
        i += 1
        input_ids1 = tokenizer(prompt1,   return_tensors='pt').input_ids.cuda()
        enforce_phrase,  next_word = False, ''

        for step in range(max_len):
            
            natural_text = tokenizer.decode(input_ids1[0]).replace( prompt1,'')
            #print(natural_text)
            
            probs, rd = predict_nextlabel_given_model(natural_text, pun_word, alter_word, label_predictor_model, label_predictor_tokenizer)
            #print(probs)

            if max(probs)>0.7:
                
                input_ids1_l, candidates, eos = generate_next_word_new(model, input_ids1, top_k, temperature = T, num_samples=5)
                if eos ==True:
                    break
                    
                
                input_ids1_l, candidates = clean(input_ids1_l, candidates)   
                #print(candidates)
                
                if rd==1:
                    score_d = get_f([re.sub(r'[^\w\s]', '', cand.lower()) for cand in candidates], pun_word)
                if rd==2:
                    score_d = get_f([re.sub(r'[^\w\s]', '', cand.lower()) for cand in candidates], alter_word)
                elif rd==0:
                    score_d1 = get_f(candidates, pun_word)
                    score_d2 = get_f(candidates, pun_word)
                    score_d = [max(value) for value in zip(score_d1,score_d2)]
                
                if len(natural_text.split())>0:
                    enforce_phrase,  next_word = check_phrase(natural_text.split()[-1],phrase)
                    #print(enforce_phrase,  next_word)
                    
                if context_word in candidates[:2]:
                    #print(f'e{rd}', end = '\t')
                    i = candidates.index(context_word)
                elif enforce_phrase and next_word in candidates:
                    #print(f'e{rd}', end = '\t')
                    i = candidates.index(next_word)
                elif rd==0:
                    #print(rd, end = '\t')
                    for i in range(len(score_d)):
                        if score_d[i] <0.2:
                            break
                else:
                    #print(rd, end = '\t')
                    i = torch.argmax(torch.tensor(score_d))
                
                input_ids1 = input_ids1_l[i]
                
                #print(tokenizer.decode(input_ids1[0], skip_special_tokens=True).split()[-1], end = '\n')
            
            else: 
                input_ids1_l, candidates, eos = generate_next_word_new(model, input_ids1, top_k, temperature = T)
                if eos:
                    break
                input_ids1 = input_ids1_l[0]
                #print('-', end = '\t')
                #print(tokenizer.decode(input_ids1[0], skip_special_tokens=True).split()[-1], end = '\n')  
        gen = tokenizer.decode(input_ids1[0], skip_special_tokens=True).replace( prompt1,'').strip('\t')
        if pun_word in gen:
            generated_puns.append(gen)
    
    return generated_puns
    

def generate_pun_vanilla_gpt3(pun_word, alter_word, phrase, context_word, num_generations):
    max_len = 35
    T = 0.9
    top_k = 30
    generated_puns = []
    prefix = '''Pun word: taxi\nAlternative word: tax \nGenerate a pun: The driver owned eighty cabs and got in all that taxi trouble due to the debt. \n\nPun word: dyed \nAlternative word: died\nGenerate a pun:   Yesterday I accidentally swallowed some food coloring -- The doctor says I'm OK, but I feel like I've dyed a little inside.\n\n\nPun word: poodles\nAlternative word: puddle\nGenerate a pun:  It was raining cats and dogs, so there were poodles all over the road.\n\nPun word: soled\nAlternative word: sold \nGenerate a pun:   The owner of the old boots had worn them for a year before he soled them for fifty dollars at the store.\n\n'''
    prompt = prefix +f'''Pun word: {pun_word}\nAlternative word: {alter_word}\nGenerate a pun: '''

    response = openai.Completion.create(engine="davinci-instruct-beta",
          prompt=prompt,
          temperature=T,
          max_tokens=max_len,
          top_p=0.95,
          n = num_generations)
                #print('-', end = '\t')
                #print(input_text.split()[-1], end = '\n')
    for r in response['choices']:
        text = r['text'].strip('\n')
        #print(text)
        generated_puns.append(text.strip())


    return generated_puns


def generate_pun_extract_gpt3(pun_word, alter_word, phrase, context_word, num_generations):
    max_len = 35
    T = 0.9
    top_k = 30
    generated_puns = []
    prefix = '''Pun word: taxi\nAlternative word: tax \nGenerate a sentence with the context word \"driver\", \"debt\" and the phrase \"got in all that taxi trouble\": The driver owned eighty cabs and got in all that taxi trouble due to the debt. \n\nPun word: dyed \nAlternative word: died\nGenerate a sentence with the context word \"coloring\", \"hospital\" and the phrase \"I've dyed a little inside\":   Yesterday I accidentally swallowed some food coloring -- The doctor says I'm OK, but I feel like I've dyed a little inside.\n\n\nPun word: poodles\nAlternative word: puddle\nGenerate a sentence with the context word \"dogs\", \"raining\" and the phrase \"poodles all over the road\":  It was raining cats and dogs, so there were poodles all over the road.\n\nPun word: soled\nAlternative word: sold \nGenerate a sentence with the context word \"boots\", \"worn\" and the phrase \"were soled for fifty dollars at the store\":   The owner of the old boots had worn them for a year before he soled them for fifty dollars at the store.\n\n'''
    prompt = prefix +f'''Pun word: {pun_word}\nAlternative word: {alter_word}\nGenerate a sentence with the context word \"{context_word}\" and the phrase \"{phrase}\": '''

    response = openai.Completion.create(engine="davinci-instruct-beta",
          prompt=prompt,
          temperature=T,
          max_tokens=max_len,
          top_p=0.95,
          n = num_generations)
                #print('-', end = '\t')
                #print(input_text.split()[-1], end = '\n')
    for r in response['choices']:
        text = r['text'].strip('\n')
        #print(text)
        generated_puns.append(text.strip())


    return generated_puns

def generate_pun_gpt3(pun_word, alter_word, phrase, context_word, label_predictor_model, label_predictor_tokenizer, num_generations):
    max_len = 35
    T = 0.9
    top_k = 30
    generated_puns = []
    prefix = '''Pun word: taxi\nAlternative word: tax \nGenerate a sentence with the context word \"driver\", \"debt\" and the phrase \"got in all that taxi trouble\": The driver owned eighty cabs and got in all that taxi trouble due to the debt. \n\nPun word: dyed \nAlternative word: died\nGenerate a sentence with the context word \"coloring\", \"hospital\" and the phrase \"I've dyed a little inside\":   Yesterday I accidentally swallowed some food coloring -- The doctor says I'm OK, but I feel like I've dyed a little inside.\n\n\nPun word: poodles\nAlternative word: puddle\nGenerate a sentence with the context word \"dogs\", \"raining\" and the phrase \"poodles all over the road\":  It was raining cats and dogs, so there were poodles all over the road.\n\nPun word: soled\nAlternative word: sold \nGenerate a sentence with the context word \"boots\", \"worn\" and the phrase \"were soled for fifty dollars at the store\":   The owner of the old boots had worn them for a year before he soled them for fifty dollars at the store.\n\n'''
    prompt = prefix +f'''Pun word: {pun_word}\nAlternative word: {alter_word}\nGenerate a sentence with the context word \"{context_word}\" and the phrase \"{phrase}\": '''

    for i in range(num_generations):
        input_text = prompt
        enforce_phrase,  next_word = False, ''

        for step in range(max_len): 
            natural_text = input_text.replace( prompt,'')
            probs, rd = predict_nextlabel_given_model(natural_text, pun_word, alter_word, label_predictor_model, label_predictor_tokenizer)
            if max(probs)>0.5:
                input_text_l, candidates, eos = generate_next_word_gpt3(input_text, T, num_samples=10)
                if eos ==True:
                    break
                if rd==1:
                    score_d = get_f([re.sub(r'[^\w\s]', '', cand.lower()) for cand in candidates], pun_word)
                if rd==2:
                    score_d = get_f([re.sub(r'[^\w\s]', '', cand.lower()) for cand in candidates], alter_word)
                elif rd==0:
                    score_d1 = get_f(candidates, pun_word)
                    score_d2 = get_f(candidates, pun_word)
                    score_d = [max(value) for value in zip(score_d1,score_d2)]
                
                if len(natural_text.split())>0:
                    enforce_phrase,  next_word = check_phrase(natural_text.split()[-1],phrase)
                    #print(enforce_phrase,  next_word)
                    
                if context_word in candidates[:2]:
                    #print(f'e{rd}', end = '\t')
                    i = candidates.index(context_word)
                elif enforce_phrase and next_word in candidates:
                    #print(f'e{rd}', end = '\t')
                    i = candidates.index(next_word)
                elif rd==0:
                    #print(rd, end = '\t')
                    for i in range(len(score_d)):
                        if score_d[i] <0.2:
                            break
                else:
                    #print(rd, end = '\t')
                    i = torch.argmax(torch.tensor(score_d))
                
                input_text = input_text_l[i]
                
                #print(input_text.split()[-1], end = '\n')
                
            
            else: 
                input_text_l, candidates, eos = generate_next_word_gpt3(input_text, T)
                if eos:
                    break
                input_text = input_text_l[0]
                #print('-', end = '\t')
                #print(input_text.split()[-1], end = '\n')
        generated_puns.append(input_text.replace( prompt,'').strip('\t'))


    return generated_puns

def generate_pun_old(model, pun_word, alter_word, phrase, context_word, label_predictor_model, label_predictor_tokenizer, num_generations=5):
    max_len = 35
    temperature = 0.9
    top_k = 30
    
    
    
    generated_puns = []
    prompt1 = 'Generate a sentence with the context word "' + context_word + '" and the phrase "' + phrase + '": '
    for i in range(num_generations):
        
        input_ids1 = tokenizer(prompt1,   return_tensors='pt').input_ids.cuda()

        for step in range(max_len):

            natural_text = tokenizer.decode(input_ids1[0]).replace( prompt1,'')

            probs, rd = predict_nextlabel_given_model(natural_text, pun_word, alter_word, label_predictor_model, label_predictor_tokenizer)

            if rd ==1 or rd ==2:
                input_ids1_l = []
                candidates = []
                for i in range(5):
                    try:
                        input1, eos = generate_next_word(model, tokenizer, input_ids1)
                    except:
                        continue

                    if eos ==True:
                        input_ids1 = input1
                        break

                    else:
                        input_ids1_l.append(input1)
                        new_natural = tokenizer.decode(input1[0]).replace( prompt1,'')
                        candidates.append(new_natural.split()[-1])
                if eos:
                    break
                if rd==1:
                    score_d = get_f(candidates, pun_word)
                elif rd==2:
                    score_d = get_f(candidates, alter_word)
                input_ids1 = input_ids1_l[i]

            else: 
                input1, eos = generate_next_word(model, tokenizer, input_ids1)
                input_ids1 = input1
                if eos:
                    break


        generated_puns.append(tokenizer.decode(input_ids1[0], skip_special_tokens=True).replace( prompt1,'').strip('\t'))
    return generated_puns
