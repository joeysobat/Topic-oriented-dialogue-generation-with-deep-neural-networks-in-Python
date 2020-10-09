
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math
import torch.utils.data as utils
from torch.autograd import Variable
import Data_processing as dpros
import pandas as pd




## this class defines the normalization operation between layers:

class Norm(nn.Module):
    def __init__(self, dim_model, eps = 1e-6):
        super().__init__()
    
        self.size = dim_model

        self.alpha = nn.Parameter(torch.ones(self.size))
        self.bias = nn.Parameter(torch.zeros(self.size))
        self.eps = eps

    def forward(self, x):
      
        norm = self.alpha * (x - x.mean(dim=-1, keepdim=True)) / (x.std(dim=-1, keepdim=True) + self.eps) + self.bias
        return norm



## this class defines the self-attention module:

class Self_attention(nn.Module):
    
  def __init__(self, dim_model, dim_k, dim_v):

    super().__init__()

    self.dim_k = dim_k
    self.dropout = nn.Dropout(0.1)
    
  def forward(self, key, query, value, mask):

    KQ_mul = torch.matmul(query, key.transpose(2, 3))

    KQ_scale = KQ_mul/math.sqrt(self.dim_k)
    mask = mask.unsqueeze(1)
    KQ_scale = KQ_scale.masked_fill(mask == 0, -1e9)

    KQ_softmax = self.dropout(F.softmax(KQ_scale, dim=-1))

    attention = torch.matmul(KQ_softmax, value)

    return attention




## this class performs the attention operation with multiple heads:

class Multi_Head_Attention(nn.Module):
    
  def __init__(self, dim_model, h_dim_v, dim_k, dim_v, num_heads):

    super().__init__()
    
    self.linear_q = nn.Linear(dim_model, dim_model, bias=False)
    self.linear_k = nn.Linear(dim_model, dim_model, bias=False)
    self.linear_v = nn.Linear(dim_model, dim_model, bias=False)

    self.num_heads = num_heads
    self.dim_model = dim_model
    self.head_dim = dim_model // num_heads
    self.linear = nn.Linear(dim_model, dim_model, bias=False)

    self.attention_layer = Self_attention(dim_model, dim_k, dim_v)
    
  def forward(self, key, query, value, mask):
    
    key_linear = self.linear_k(key).view(key.size(0), -1, self.num_heads, int(self.head_dim))
    query_linear = self.linear_q(query).view(key.size(0), -1, self.num_heads, int(self.head_dim))
    value_linear = self.linear_v(value).view(key.size(0), -1, self.num_heads, int(self.head_dim))
       
    key_linear = key_linear.transpose(1,2)
    query_linear = query_linear.transpose(1,2)
    value_linear = value_linear.transpose(1,2)

    attention = self.attention_layer(key_linear, query_linear, value_linear, mask)

    attention_concat = attention.transpose(1, 2).contiguous().view(key.size(0), -1, self.dim_model)

    attention_result = self.linear(attention_concat)

    return attention_result




## this class defines the encoder layer, with the respective normalization, multi-head attention
## and feed forward layers:

class Trans_encoder_layer(nn.Module):
    
  def __init__(self, dim_model, h_dim_v, dim_k, dim_v, dim_ff, num_heads):

    super().__init__()

    self.linear_feed1 = nn.Linear(dim_model, dim_ff)
    self.linear_feed2 = nn.Linear(dim_ff, dim_model)

    self.norm_1 = Norm(dim_model)
    self.norm_2 = Norm(dim_model)

    self.dropout_1 = nn.Dropout(0.1)
    self.dropout_2 = nn.Dropout(0.1)

    self.attention_layer = Multi_Head_Attention(dim_model, h_dim_v, dim_k, dim_v, num_heads)
    
  def forward(self, input, input_mask):

    input_norm = self.norm_1(input)

    attention = self.dropout_1(self.attention_layer(input_norm, input_norm, input_norm, input_mask))

    add_and_norm1 = attention + input

    input_norm_2 = self.norm_2(add_and_norm1)

    feed_forward_res = self.dropout_2(self.linear_feed2(F.relu(self.linear_feed1(input_norm_2))))

    add_and_norm2 = feed_forward_res + add_and_norm1

    return add_and_norm2




## this class defines the decoder layer, with the respective normalization, multi-head attention
## and feed forward layers:

class Trans_decoder_layer(nn.Module):
    
  def __init__(self, dim_model, h_dim_v, dim_k, dim_v, dim_ff, num_heads):

    super().__init__()

    self.linear_feed1 = nn.Linear(dim_model, dim_ff)
    self.linear_feed2 = nn.Linear(dim_ff, dim_model)

    self.norm_1 = Norm(dim_model)
    self.norm_2 = Norm(dim_model)
    self.norm_3 = Norm(dim_model)

    self.dropout_1 = nn.Dropout(0.1)
    self.dropout_2 = nn.Dropout(0.1)
    self.dropout_3 = nn.Dropout(0.1)

    self.attention_layer_1 = Multi_Head_Attention(dim_model, h_dim_v, dim_k, dim_v, num_heads)
    self.attention_layer_2 = Multi_Head_Attention(dim_model, h_dim_v, dim_k, dim_v, num_heads)
    
  def forward(self, input, enc_outputs, input_mask, target_mask):

    input_norm = self.norm_1(input)

    attention1 = self.dropout_1(self.attention_layer_1(input_norm, input_norm, input_norm, target_mask))

    add_and_norm1 = attention1 + input

    input_norm_2 = self.norm_2(add_and_norm1)

    attention2 = self.dropout_2(self.attention_layer_2(enc_outputs, input_norm_2, enc_outputs, input_mask))

    add_and_norm2 = attention2 + add_and_norm1

    input_norm_3 = self.norm_3(add_and_norm2)

    feed_forward_res = self.dropout_3(self.linear_feed2(F.relu(self.linear_feed1(input_norm_3))))

    add_and_norm3 = feed_forward_res + add_and_norm2

    return add_and_norm3
    
    


## this class defines the full encoder, with the positional encoding and the six encoder layers:

class Trans_encoder(nn.Module):

    def __init__(self, input_dim, dim_model, h_dim_v, dim_k, dim_v, dim_ff, seq_len, num_heads, num_layers):
    
        super().__init__()
        
        self.dim_model = dim_model
        self.seq_len = seq_len
        self.num_layers = num_layers
        
        encoder_layer = Trans_encoder_layer(dim_model, h_dim_v, dim_k, dim_v, dim_ff, num_heads)
        self.enc_layers = dpros._get_clones(encoder_layer, num_layers)
        self.norm = Norm(dim_model)
        self.dropout = nn.Dropout(0.1)
        
        pos_encoding = torch.zeros(seq_len, dim_model).detach()

        for pos in range(seq_len):
            for i in range(0, dim_model, 2):

                pos_encoding[pos, i] = math.sin(pos / (10000 ** ((2 * i) / dim_model)))
                pos_encoding[pos, i + 1] = math.cos(pos / (10000 ** ((2 * (i + 1)) / dim_model)))
            
        pos_encoding = pos_encoding.unsqueeze(0)
        self.register_buffer('pos_encoding', pos_encoding)
        
    def forward(self, input, input_mask):
    
        embed_input = input * math.sqrt(self.dim_model)
        enc_sentence = self.dropout(embed_input + self.pos_encoding[:,:self.seq_len].cuda())
        
        for i in range(self.num_layers):

            enc_sentence = self.enc_layers[i](enc_sentence, input_mask)
            
        enc_sentence = self.norm(enc_sentence)
        
        return enc_sentence
        



## this class defines the full decoder, with the positional encoding and the six decoder layers:

class Trans_decoder(nn.Module):

    def __init__(self, input_dim, dim_model, h_dim_v, dim_k, dim_v, dim_ff, seq_len, num_heads, num_layers):
    
        super().__init__()
        
        self.dim_model = dim_model
        self.num_layers = num_layers
        
        decoder_layer = Trans_decoder_layer(dim_model, h_dim_v, dim_k, dim_v, dim_ff, num_heads)
        self.dec_layers = dpros._get_clones(decoder_layer, num_layers)
        self.norm = Norm(dim_model)
        self.dropout = nn.Dropout(0.1)
        
        pos_encoding = torch.zeros(seq_len, dim_model).detach()

        for pos in range(seq_len):
            for i in range(0, dim_model, 2):

                pos_encoding[pos, i] = math.sin(pos / (10000 ** ((2 * i) / dim_model)))
                pos_encoding[pos, i + 1] = math.cos(pos / (10000 ** ((2 * (i + 1)) / dim_model)))
            
        pos_encoding = pos_encoding.unsqueeze(0)
        self.register_buffer('pos_encoding', pos_encoding)
        
    def forward(self, target, enc_sentence, input_mask, target_mask):
    
        target_length = target.size(1)
        embed_target = target * math.sqrt(self.dim_model)
        dec_sentence = self.dropout(embed_target + self.pos_encoding[:,:target_length].cuda())
        
        for i in range(self.num_layers):

            dec_sentence = self.dec_layers[i](dec_sentence, enc_sentence, input_mask, target_mask)
            
        dec_sentence = self.norm(dec_sentence)
        
        return dec_sentence
        
        