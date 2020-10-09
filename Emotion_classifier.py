

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import torch.utils.data as utils
from torch.autograd import Variable
import Data_processing_classifier as dpros
import matplotlib.pyplot as plt
import numpy as np
import itertools
        
        
        
## this class defines the self-attention module:
        
class SelfAttention(nn.Module):
    def __init__(self, hid_dim, dropout):
        super(SelfAttention, self).__init__()

        self.drop_1 = nn.Dropout(dropout)
        self.drop_2 = nn.Dropout(dropout)
        self.drop_3 = nn.Dropout(dropout)
        self.tanh = nn.Tanh()
        self.hid_dim = hid_dim

    def forward(self, decoder_out, final_hidden):

        decoder_out = self.drop_1(decoder_out.permute(1, 0, 2)) ## decoder_out.size() = (batch size, sequence length, hidden size * 2)
        final_hidden = final_hidden.permute(1, 0, 2) ## final_hidden.size() = (batch size, num directions, hidden size)
        final_hidden = final_hidden.contiguous().view(-1, self.hid_dim * 2) ## final_hidden.size() = (batch size, num directions * hidden size)

        attn_weights = torch.bmm(decoder_out, final_hidden.unsqueeze(2)).squeeze(2)

        soft_attn_weights = self.drop_2(F.softmax(attn_weights, 1))

        new_hidden_state = self.drop_3(self.tanh(torch.bmm(decoder_out.transpose(1, 2), soft_attn_weights.unsqueeze(2)).squeeze(2)))

        return new_hidden_state
    
    
    
## this function is used to show the confusion matrix when testing the accuracy of the classifier:

def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix')

    print(cm)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    
    
    
    
