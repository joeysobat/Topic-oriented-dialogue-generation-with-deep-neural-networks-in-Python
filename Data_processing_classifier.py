
import torch
import nltk
import itertools
import numpy as np
from torch.utils.data import Dataset, DataLoader


VOCAB_SIZE = 25000

BATCH_SIZE = 32

BLACKLIST = '!^_:;+,-.?'

limit = {'maxs': 50, 'mins': 0}

UNK = 'unk'



## this function splits a line when a comma is found, dividing the sentence in parts:

def split_line(line):
    return line.split(',')
    
    

## this function returns in a list all the lines of a file:

def read_lines(filename):
    return open(filename).read().split('\n')[:-1]
    
    


## this function filters punctuation characters, contractions, and characters that are on the blacklist:

def filter_line(line, stopwords_dict, contractions):
  new_line_cinema = line.lower()
  new_line_cinema = new_line_cinema.replace("!", "")
  new_line_cinema = new_line_cinema.replace("+", "")
  new_line_cinema = new_line_cinema.replace(",", "")
  new_line_cinema = new_line_cinema.replace("...", "")
  for word in new_line_cinema.split():
    if word in contractions:
        new_line_cinema = new_line_cinema.replace(word, contractions[word])
  return new_line_cinema
  
  

## this function returns all the lines which are shorter than the maximum length:

def filter_data(sequences, targets):
    
    filtered_q, filtered_a = [], []
    raw_data_len = len(sequences)

    for i in range(0, len(sequences)):
        qlen = len(sequences[i].split(' '))
        if qlen >= limit['mins'] and qlen <= limit['maxs']:
          filtered_q.append(sequences[i])
          filtered_a.append(targets[i])

    return filtered_q, filtered_a
    
    

## this function creates the index-to-word and word-to-index dictionaries, based on the
## frequency distribution:

def index_(tokenized_sentences, vocab_size):
    
    freq_dist = nltk.FreqDist(itertools.chain(*tokenized_sentences))

    vocab = freq_dist.most_common(vocab_size)

    index2word = ['_'] + [UNK] + [ x[0] for x in vocab ]

    word2index = dict([(w,i) for i,w in enumerate(index2word)] )

    return index2word, word2index, freq_dist
    
    
    

## this function returns an array of the indices for the words of the sentences:

def zero_pad(stokenized, w2idx):

    data_len = len(stokenized)

    idx_s = np.zeros([data_len, limit['maxs']], dtype=np.int32)

    for i in range(data_len):
        s_indices = pad_seq(stokenized[i], w2idx, limit['maxs'])

        idx_s[i] = np.array(s_indices)

    return idx_s
    
    
    

## this function makes the actual conversion of words into indices, and adds the padding
## token until the sentence reaches the maximum length:

def pad_seq(seq, lookup, maxlen):

    indices = []
    
    for word in seq:
        if word in lookup:
            indices.append(lookup[word])
        else:
            indices.append(lookup[UNK])

    return indices + [0]*(maxlen - len(seq))
    
    
    


## this function returns the list of words from a list of indices:

def decode(sequence, lookup, separator=''):
    return separator.join([lookup[element] for element in sequence if element])
    
    
    
