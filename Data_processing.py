
import torch
import torch.nn.functional as F
import nltk
import itertools
import re
import numpy as np
import copy
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader


EN_WHITELIST = '0123456789abcdefghijklmnopqrstuvwxyz.,?- '

EN_BLACKLIST = '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~\''

limit = {'maxq': 30, 'minq': 0, 'maxa': 30, 'mina': 0}

UNK = 'unk'

batch_size = 256

VOCAB_SIZE = 25000
	

## this function returns a list of all lines from a file:

def read_lines(filename):
    return open(filename).read().split('\n')[:-1]
    
   
   
## this function only returns the characters on a line that are in the whitelist,
## and so removing the ones that are not in it:

def filter_line(line, whitelist):
    return ''.join([ ch for ch in line if ch in whitelist ])
    
    
    
## the same as "filter_line", but there is no whitelist:
    
def filter_line_opensub(line):
    return ''.join([ch for ch in line])
    
    
## this function filters too long sentences. That is, it outputs the list of all
## sentences that are shorter than the maximum length (in our case, 30):

def filter_data(sequences):
    
    filtered_q, filtered_a = [], []

    for i in range(0, len(sequences), 2):
        qlen, alen = len(sequences[i].split(' ')), len(sequences[i+1].split(' '))
        if qlen > limit['minq'] and qlen < limit['maxq']:
            if alen > limit['mina'] and alen < (limit['maxa'] - 1):
                filtered_q.append(sequences[i])
                filtered_a.append(sequences[i+1])

    return filtered_q, filtered_a
    
    
## the same as "filter_data_opensub", but specific for the sentences of the opensub
## dataset:
    
def filter_data_opensub(sequences_source, sequences_target):
    
    filtered_q, filtered_a = [], []

    for i in range(0, len(sequences_source)):
        qlen, alen = len(sequences_source[i].split(' ')), len(sequences_target[i].split(' '))
        if qlen > limit['minq'] and qlen < limit['maxq']:
            if alen > limit['mina'] and alen < limit['maxa']:
                filtered_q.append(sequences_source[i])
                filtered_a.append(sequences_target[i])

    return filtered_q, filtered_a
    
    
    
## this function creates the dictionaries word-to-index and index-to-word for the
## Twitter database, based on the frequency distribution, and then adds the emotion tokens:
    
def index_(tokenized_sentences, vocab_size):
    
    freq_dist = nltk.FreqDist(itertools.chain(*tokenized_sentences))

    vocab = freq_dist.most_common(vocab_size)

    index2word = {}
    word2index = {}

    index2word = ['<pad>'] + [UNK] + [ x[0] for x in vocab ] + ['<s>'] + ['</s>']

    word2index = dict([(w, i) for i, w in enumerate(index2word)])
    index2word = dict([(i, w) for i, w in enumerate(index2word)])
    
    new_vocab_size = len(index2word)
    
    for i in range(9):
        emotion = '<' + str(i) + '>'
        word2index[emotion] = new_vocab_size
        index2word[new_vocab_size] = emotion
        new_vocab_size += 1

    return index2word, word2index
    
    
    
    
## this function creates the dictionaries word-to-index and index-to-word for the
## Opensub database, adding the emotion tokens:
    
def index_opensub():

    word2id = {}
    id2word = {}
    
    id2word[0] = '<pad>'
    word2id['<pad>'] = 0

    with open('movie_25000.txt') as file:
    
        all_lines = file.readlines()
        for i, line in enumerate(all_lines):
        
            id2word[i+1] = line.rstrip('\n')
            word2id[line.rstrip('\n')] = i+1

    vocab_size = len(id2word)
    id2word[vocab_size] = '<s>'
    word2id['<s>'] = vocab_size
    id2word[vocab_size+1] = '</s>'
    word2id['</s>'] = vocab_size+1
    
    new_vocab_size = len(id2word)
    
    for i in range(9):
        emotion = '<' + str(i) + '>'
        word2id[emotion] = new_vocab_size
        id2word[new_vocab_size] = emotion
        new_vocab_size += 1
    
    return word2id, id2word
    
    
    
## this function outputs the source and target sentences as lists of indices and adding zero-
## padding to make all elements the same length, as well as the list of actual lengths of the
## source sentences:

def zero_pad(qtokenized, atokenized, w2idx):

    data_len = len(qtokenized)

    idx_q = np.zeros([data_len, limit['maxq']], dtype=np.int32) 
    idx_a = np.zeros([data_len, limit['maxa']], dtype=np.int32)
    q_len = np.zeros([data_len], dtype=np.int32)

    for i in range(data_len):
        q_indices, q_lengths = pad_seq(qtokenized[i], w2idx, limit['maxq'])
        a_indices = pad_seq_target(atokenized[i], w2idx, limit['maxa'])

        idx_q[i] = np.array(q_indices)
        idx_a[i] = np.array(a_indices)
        q_len[i] = q_lengths

    return idx_q, idx_a, q_len
    
    
    
## the same as for the "zero_pad", but in the case of the transformer we don't need the
## lengths of the source sentences:

def zero_pad_trans(qtokenized, atokenized, w2idx):

    data_len = len(qtokenized)

    idx_q = np.zeros([data_len, limit['maxq']], dtype=np.int32) 
    idx_a = np.zeros([data_len, limit['maxa']], dtype=np.int32)

    for i in range(data_len):
        q_indices = pad_seq_trans(qtokenized[i], w2idx, limit['maxq'])
        a_indices = pad_seq_target(atokenized[i], w2idx, limit['maxa'])

        idx_q[i] = np.array(q_indices)
        idx_a[i] = np.array(a_indices)

    return idx_q, idx_a
    


## same as in "zero_pad", but only for the source sentences:

def zero_pad_source(tokenized, w2idx):
    
    data_len = len(tokenized)

    idx = np.zeros([data_len, limit['maxq']], dtype=np.int32)
    lengths = np.zeros([data_len], dtype=np.int32)

    for i in range(data_len):
        indices, lengths_i = pad_seq(tokenized[i], w2idx, limit['maxq'])

        idx[i] = np.array(indices)
        lengths[i] = lengths_i

    return idx, lengths



## same as in "zero_pad", but only for the target sentences:

def zero_pad_target(tokenized, w2idx):

    data_len = len(tokenized)

    idx = np.zeros([data_len, limit['maxa']], dtype=np.int32)

    for i in range(data_len):
        indices = pad_seq_target(tokenized[i], w2idx, limit['maxa'])

        idx[i] = np.array(indices)

    return idx
    
    
    
'''
 replace words with indices in a sequence
  replace with unknown if word not in lookup
    return [list of indices]
'''

## the function that actually replaces the words with indices. If the
## word is unknown, it is replaced with the UNK token. The sentence is
## filled with zeros until it reaches the maximum length. It also outputs
## the length of the sentence:

def pad_seq(seq, lookup, maxlen):

    indices = []
    
    for word in seq:
        if word in lookup:
            indices.append(lookup[word])
        else:
            indices.append(lookup[UNK])

    return indices + [0]*(maxlen - len(seq)), len(indices)
    
    
    
## the same as "pad_seq", but without outputing the length of the sentence:

def pad_seq_trans(seq, lookup, maxlen):

    indices = []
    
    for word in seq:
        if word in lookup:
            indices.append(lookup[word])
        else:
            indices.append(lookup[UNK])

    return indices + [0]*(maxlen - len(seq))
    
    
    
## like in "pad_seq", but for the target sentences. That is, we also add the "start-
## of-sequence" and "end-of-sequence" tokens to the sentence:
    
def pad_seq_target(seq, lookup, maxlen):

    indices = []

    if len(seq) > maxlen - 2:
      seq = seq[:maxlen - 2]
    
    for word in seq:
        if word in lookup:
            indices.append(lookup[word])
        else:
            indices.append(lookup[UNK])

    return [lookup['<s>']] + indices + [lookup['</s>']] + [0]*(maxlen - len(seq) - 2)
    
    
    
    
## this function adds the respective emotion tokens to their source sentences, and the
## lengths of the sentences are added 1:
    
def emotion_pad(qtokenized, q_lengths, emotion, w2idx):

    data_len = len(qtokenized)

    idx_q = np.zeros([data_len, limit['maxq']], dtype=np.int32)
    q_len = np.zeros([data_len], dtype=np.int32)

    for i in range(data_len):
    
        if len(qtokenized[i]) > limit['maxq'] - 1:
            processed_q = qtokenized[i][:limit['maxq'] - 1]
        else:
            processed_q = qtokenized[i]
    
        if isinstance(emotion, list):
          tag = w2idx['<' + str(emotion[i]) + '>']
        else:
          tag = w2idx['<' + str(emotion) + '>']
          
        idx_q[i] = np.concatenate([np.array([tag]), processed_q])
        q_len[i] = q_lengths[i] + 1

    return idx_q, q_len
    
    
    
## the same as "emotion_pad", but for the transformer, without the lengths of the sentences:
    
def emotion_pad_trans(qtokenized, emotion, w2idx):

    data_len = len(qtokenized)
    idx_q = np.zeros([data_len, limit['maxq']], dtype=np.int32)

    for i in range(data_len):
    
        if len(qtokenized[i]) > limit['maxq'] - 1:
            processed_q = qtokenized[i][:limit['maxq'] - 1]
        else:
            processed_q = qtokenized[i]
    
        if isinstance(emotion, list):
          tag = w2idx['<' + str(emotion[i]) + '>']
        else:
          tag = w2idx['<' + str(emotion) + '>']
          
        idx_q[i] = np.concatenate([np.array([tag]), processed_q])

    return idx_q
    
    

## this function transforms a sequence of indices into a sentence of words:

def decode(sequence, lookup, separator=''):
    return separator.join([ lookup[element] for element in sequence])
    
    
    
## this function returns a copy of a layer N times:
    
def _get_clones(module, N):
    return torch.nn.ModuleList([copy.deepcopy(module) for i in range(N)])
    
    
    
## this function creates the input and target masks for the transformer:
    
def create_masks(input, target, word2id):

    input_pad = word2id['<pad>']
    input_mask = (input != input_pad).unsqueeze(1)

    target_pad = word2id['<pad>']
    target_mask = (target != target_pad).unsqueeze(1)

    size = target.size(1)
    nopeak_mask = np.triu(np.ones((1, size, size)), k=1).astype('uint8')
    nopeak_mask = Variable(torch.from_numpy(nopeak_mask) == 0)
    target_mask = target_mask.cuda()
    nopeak_mask = nopeak_mask.cuda()
    target_mask = target_mask & nopeak_mask

    return input_mask, target_mask
    
    