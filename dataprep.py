import numpy as np
import os
from collections import Counter
import torch
from torch.utils.data import TensorDataset, DataLoader

def load_data(path):
    text_file = os.path.join(path)
    with open(text_file,'r') as f:
        data = f.read()
    return data

def word_embeddings(text):
    char_counts = Counter(text)
    char_counts = sorted(char_counts,key=char_counts.get,reverse=True)
    vocab_to_int = {char_counts[x]:x for x in range(len(char_counts))}
    int_to_vocab = {x:char_counts[x] for x in range(len(char_counts))}
    return vocab_to_int,int_to_vocab

def punctuation_handler(text,for_gen=False):
    punctuation = {'.':'<period>',',':'<comma>','"':'<quotationmark>',';':'<semicolon>','!':'<exclamation>','?':'<question>',
           '(':'<leftparentheses>',')':'<rightparentheses>','-':'<dash>','\n':'<newline>'}
    for k in punctuation.items():
        if for_gen==False:
            text = text.replace(k[0],' '+k[1]+' ')
        else:
            text = text.replace(k[1],k[0])
    return text

def data_batcher(text_nums,seq_length,batch_size):
    features=[]
    targets=[]
    for x in range(0,len(text_nums)-seq_length):
        features.append(text_nums[x:x+seq_length])
        targets.append(text_nums[x+seq_length])
    features = torch.from_numpy(np.array(features))
    targets = torch.from_numpy(np.array(targets))
    data = TensorDataset(features,targets)
    data_loader = DataLoader(data,batch_size=batch_size,shuffle=True)
    return data_loader

def data_processor(dir):
    text = load_data(dir)
    text = punctuation_handler(text)
    text = text.lower()
    text = text.split()
    v_to_i,i_to_v = word_embeddings(text)
    text_nums = [v_to_i[word] for word in text]
    return v_to_i,i_to_v,text_nums

if __name__ == '__main__':
    data_dir = './data/lyrics.txt'
    text = load_data(data_dir)

    print ('Length of dataset: {}'.format(len(text)))
    text = punctuation_handler(text)
    print(text[:100])

    text=text.lower()
    text=text.split()

    v_to_i,i_to_v = word_embeddings(text)

    text_nums = [v_to_i[word] for word in text]
    print(text_nums[:100])