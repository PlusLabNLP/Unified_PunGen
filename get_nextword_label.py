import torch
import torch.nn as nn
import transformers
from transformers import AutoModelForSequenceClassification, AutoTokenizer

predictor_model_checkpoint = 'label_predictor/checkpoint'

device='cuda'

tokenizer = AutoTokenizer.from_pretrained(predictor_model_checkpoint)
model = AutoModelForSequenceClassification.from_pretrained(predictor_model_checkpoint).to(device)

model = model.eval()


def predict_nextlabel(sent, pun_word, alter_word, device='cuda'):
    softmax = nn.Softmax(dim=1)
    #label_dict = {0: 'A', 1: 'D_F1', 2: 'D_F2'}
    
    input_sent = 'Pun Sentence: ' + sent + ' [SEP] Pun word: ' + pun_word + ' [SEP] Alternative Word: ' + alter_word
    #print(input_sent)
    input_ids = torch.tensor([tokenizer(input_sent)['input_ids']]).to(device)
    attention_mask = torch.tensor([tokenizer(input_sent)['attention_mask']]).to(device)
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    probs = softmax(outputs.logits).detach().to('cpu')[0]
    label = int(torch.argmax(probs).detach().to('cpu'))
    return probs.tolist(), label
