

import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
import pandas as pd

import torch.utils.data as utils
from torch.autograd import Variable
import Data_processing as dpros



## the "Encoder" class defines the encoder module with an LSTM layer. It also performs a special
## operation to pack the sentences in order to reduce the computation load:

class Encoder(nn.Module):
  def __init__(self, input_dim, emb_dim, hid_dim, dropout, num_layers=2):
    super().__init__()

    self.hid_dim = hid_dim
    self.num_layers = num_layers
        
    self.rnn = nn.LSTM(emb_dim, hid_dim, num_layers, bidirectional=True, dropout=dropout)
    
  def forward(self, src, src_length):

    src = src.permute(1, 0, 2)
    
    h_0 = Variable(torch.zeros(self.num_layers * 2, src.size(1), self.hid_dim), requires_grad=False).cuda()
    c_0 = Variable(torch.zeros(self.num_layers * 2, src.size(1), self.hid_dim), requires_grad=False).cuda()

    packed = nn.utils.rnn.pack_padded_sequence(src, src_length.clamp(max=30), enforce_sorted=False)
    
    outputs, (hidden, cell) = self.rnn(packed, (h_0, c_0))

    outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs)

    hidden = torch.cat((hidden[-1], hidden[-2]), 1)
    cell = torch.cat((cell[-1], cell[-2]), 1)

    return outputs, (hidden, cell)
    
    

## this class defines the global attention module:
    
class Attention(nn.Module):
    def __init__(self, dim, dropout):
        super(Attention, self).__init__()
        
        self.linear_in = nn.Linear(dim, dim, bias=False)
        self.softmax = nn.Softmax(dim=1)
        self.linear_out = nn.Linear(dim * 2, dim, bias=False)

    def forward(self, hidden, enc_outputs):

        decode_output = self.linear_in(hidden).unsqueeze(2)

        global_alignment_weights = self.softmax(torch.bmm(enc_outputs, decode_output))
        global_alignment_weights = global_alignment_weights.view(global_alignment_weights.size(0), 1, global_alignment_weights.size(1)) # batch x 1 x sourceL
        
        context_vector = torch.bmm(global_alignment_weights, enc_outputs).squeeze(1)
        h_tilde = self.linear_out(torch.cat((context_vector, hidden), 1))

        return h_tilde
        
   

## this class defines the custom LSTM cell that will be used in the decoder module:
   
class LSTMCell(nn.Module):

    def __init__(self, input_size, hidden_size):
        super(LSTMCell, self).__init__()
        
        self.input_weights = nn.Linear(input_size, 4 * hidden_size)
        self.hidden_weights = nn.Linear(hidden_size, 4 * hidden_size)
        
    def forward(self, input, hidden):

        hx, cx = hidden
        gates = self.input_weights(input) + self.hidden_weights(hx)
        in_gate, forget_gate, cell_gate, out_gate = gates.chunk(4, 1)

        in_gate = torch.sigmoid(in_gate)
        forget_gate = torch.sigmoid(forget_gate)
        out_gate = torch.sigmoid(out_gate)
        cell_gate = torch.tanh(cell_gate)

        cy = (forget_gate * cx) + (in_gate * cell_gate)
        hy = out_gate * torch.tanh(cy)

        return (hy, cy)
        



## the decoder module, which performs the LSTM and attention layers at each
## word of the sentences. The results are concatenated at the end:

class Decoder(nn.Module):

    def __init__(self, vocab_size, input_size, hidden_size, dropout):
        super(Decoder, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.LSTM = LSTMCell(input_size, hidden_size)
        self.attention_layer = Attention(hidden_size, dropout)

    def forward(self, target, hidden, enc_outputs):

        output = []
        target = target.transpose(0, 1)

        for i in range(target.size(0)):

            embedded_step = target[i]
            (h_t, c_t) = self.LSTM(embedded_step, hidden)
            h_tilde = self.attention_layer(h_t, enc_outputs)
            output.append(h_tilde)
            hidden = (h_tilde, c_t)
        
        output_concat = torch.cat(output, dim=0)
        output = output_concat.view(target.size(0), target.size(1), self.hidden_size).transpose(1, 0)

        return output, (h_tilde, c_t)
        
        