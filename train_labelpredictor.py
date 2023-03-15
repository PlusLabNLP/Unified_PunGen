import os
import logging
import pandas as pd
import numpy as np

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score, classification_report
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import train_test_split
import torch
from simpletransformers.classification import ClassificationModel, ClassificationArgs


def merge_dev_plus_test_data(dev_data_path, test_data_path):
    dev_data = pd.read_csv(dev_data_path)
    print("Dev data size = ", len(dev_data))
    test_data = pd.read_csv(test_data_path)
    print("Test data size = ", len(test_data))
    frames = [dev_data, test_data]
    result = pd.concat(frames)
    result = result.sample(frac=1).reset_index(drop=True)
    print("Total data size = ", len(result))
    return result

def get_toremove_annotated():
    to_remove_annotated = []
    with open('PATH/get_A_D/jupyter-notebooks/annotated_pun_sents.txt', 'r') as f1:
        lines = f1.readlines()
        for line in lines:
            to_remove_annotated.append(line[:-1])
    return to_remove_annotated

        
def remove_annotated_ones(train_data, to_remove_annotated):
    drop_list = []
    for i in range(len(train_data)):
        sent_here = train_data.at[i, 'pun_sent']
        if sent_here in to_remove_annotated:
            drop_list.append(i)
    train_data_new = train_data.drop(index=drop_list)
    train_data_new.reset_index(inplace=True, drop=True)
    print('Dropped Annotated # =', len(drop_list))
    return train_data_new

# train_data.loc[0]
def remove_null_and_short_nextwords(data):
    drop_list = []
    for i in range(len(data)):
        if pd.isnull(data.at[i, 'next_word']) or len(data.at[i, 'next_word']) < 3:
            drop_list.append(i)
    data_new = data.drop(index=drop_list)
    data_new.reset_index(inplace=True, drop=True)
    return data_new

def get_bert_input(train_data, model_type='bert'):
    if model_type == 'bert' or model_type == 'distilbert':
        sep_token = ' [SEP] '
    elif model_type == 'roberta':
        sep_token = ' </s> '
    for i in range(len(train_data)):
#         try:
        train_data.at[i, 'pun_sent'] = train_data.at[i, 'pun_sent'] + sep_token + train_data.at[i, 'pun_word'] + sep_token + train_data.at[i, 'alter_word']
#         except:
#             print(i)
    return train_data

def drop_blank_labels(eval_data_newannotated):
    blank_list = []
    for i in range(len(eval_data_newannotated)):
        if pd.isnull(eval_data_newannotated.at[i, 'combined_label']):
            blank_list.append(i)
    eval_data_newannotated_dropped = eval_data_newannotated.drop(index=blank_list)
    eval_data_newannotated_dropped.reset_index(inplace=True, drop=True)
    return eval_data_newannotated_dropped

def merge_eval_plus_neweval_data(eval_data, eval_data_newannotated):
    print("Eval data size = ", len(eval_data))
    print("Eval data newannotated size = ", len(eval_data_newannotated))
    frames = [eval_data, eval_data_newannotated]
    result = pd.concat(frames)
    result = result.sample(frac=1).reset_index(drop=True)
    print("Total data size = ", len(result))
    return result


def get_training_results(training_log, eval_results):
    training_loss = []
    training_accuracy = []
    eval_loss = []
    eval_accuracy = []
    for i in range(len(eval_results)):
        training_accuracy.append(round(eval_results[i]['acc'], 4))
    for i in range(len(training_log)):
        training_loss.append(round(training_log[i]['train_loss'][0], 4))
        eval_loss.append(round(training_log[i]['eval_loss'][0], 4))
        eval_accuracy.append(round(training_log[i]['acc'][0], 4))
    
    print('Training Loss = ', training_loss)
    print('Training Accuracy = ', training_accuracy)
    print('Dev Loss = ', eval_loss)
    print('Dev Accuracy = ', eval_accuracy)


def train_model(model_type, hfname, num_epochs=10):
    
    model_type = model_type
    dev_data_path = 'PATH/semeval_devdata_processed_labellingapproach2.csv'
    test_data_path = 'PATH/semeval_testdata_processed_labellingapproach2.csv'
    train_data = merge_dev_plus_test_data(dev_data_path, test_data_path)
    
    to_remove_annotated = get_toremove_annotated()
    train_data = remove_annotated_ones(train_data, to_remove_annotated)
    train_data = remove_null_and_short_nextwords(train_data)
    print("Total train data size = ", len(train_data))
    train_data_new = get_bert_input(train_data, model_type=model_type)
    annotated_data_path = 'PATH/semeval_annotateddata_processed.csv'
    eval_data = get_bert_input(pd.read_csv(annotated_data_path), model_type=model_type)
    newannotated_data_path = 'PATH/semeval_additional_toannotate_manuallycombinedlabels.csv'
    eval_data_newannotated = pd.read_csv(newannotated_data_path)
    eval_data_newannotated = get_bert_input(drop_blank_labels(eval_data_newannotated))
    
    eval_data_new = merge_eval_plus_neweval_data(eval_data, eval_data_newannotated)

    train_data_new.rename(columns={'pun_sent': 'text', 'labels': 'labels'}, inplace=True)
    eval_data_new.rename(columns={'pun_sent': 'text', 'combined_label': 'labels'}, inplace=True)

    logging.basicConfig(level=logging.INFO)
    transformers_logger = logging.getLogger("transformers")
    transformers_logger.setLevel(logging.WARNING)

    train_data_new, dev_data_new = train_test_split(train_data_new,train_size=0.9, shuffle=True, random_state=42) 

    class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(train_data_new['labels']), y=list(train_data_new['labels']))
    print("Class Weights = ", list(class_weights))
    
    
    
    model_args = ClassificationArgs()
    model_args.num_train_epochs = 1
    # model_args.evaluate_generated_text = True
    model_args.evaluate_during_training = True
    model_args.evaluate_during_training_steps = 276
    model_args.evaluate_during_training_verbose = True
    model_args.manual_seed = 42
    model_args.train_batch_size = 16
    model_args.eval_batch_size = 16
    model_args.do_lower_case = True
    model_args.labels_list = ['A', 'D_F1', 'D_F2']
    model_args.overwrite_output_dir = True
    model_args.output_dir = 'outputs_' + model_type + '/'
    model_args.learning_rate = 2e-5
    
    model = ClassificationModel(model_type, hfname, num_labels=3, use_cuda=True, weight=list(class_weights), args=model_args)
    
    training_log = []
    eval_results = []
    for i in range(num_epochs):
        print('\n\nRunning Epoch', i + 1, '--')
        _, training_details = model.train_model(train_data_new, eval_df=dev_data_new, acc=accuracy_score, verbose=True)
        training_log.append(training_details)
        print("Train set accuracy after epoch", i + 1, '--')
        result, logits, wrong = model.eval_model(train_data_new, acc=accuracy_score)
        eval_results.append(result)
    print("Test set accuracy after training --")
    test_result, test_logits, test_wrong = model.eval_model(eval_data_new, acc=accuracy_score)
    
    print('Model type = ' + model_type + ' | ' + hfname)
    print('Number of epochs = ' + str(num_epochs))
    get_training_results(training_log, eval_results)


