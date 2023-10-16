import os
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchtext.data.functional import to_map_style_dataset
from torch.nn.functional import pad
import numpy as np
from matplotlib import pyplot as plt
import random
import argparse
import time
import math
from torch import Tensor

import spacy
import gensim
from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec
import nltk
import re
import torchtext.vocab as t_vocab
import torch
import wandb
import torch.nn.functional as F
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from torchtext.vocab import build_vocab_from_iterator
import pandas as pd
import random
random.seed(32)
from torch.nn.utils.rnn import pad_sequence

from Trainer import Trainer_jayaram
from Trainer_transformer import Trainer_transformer_jayaram

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# wandb setup
number = 2
NAME = "model_dropout_02_" + str(number)
ID = 'Transformer_training_' + str(number)
run = wandb.init(project='Transformer_training', name = NAME, id = ID)

def data_process(sentences, spacy_model, index: int = 0):
    #tokenized data converted to index based on vocabulary   
    data = []
    vocab = vocab_src if index == 0 else vocab_trg
    # loop through each sentence pair
    for i, sentence in enumerate(sentences):
        # tokenize the sentence and convert each word to an integers
        tensor_ = torch.tensor([vocab[token.text.lower()] for token in spacy_model.tokenizer(sentence)], dtype=torch.long)
        # append tensor representations
        data.append(tensor_)
    return data

def yield_tokens(data_iter, tokenizer):
  """
    Return the tokens for the appropriate language.  
  """
  for sentence in data_iter:
    yield tokenizer(sentence)

def load_text_data(path):

    with open(path, "r", encoding="utf-8") as file:
        data = [line.strip() for line in file]
    return data
    
def split_into_words(text):
        text = text.lower()
        text = text.strip('\n')
        text = re.sub(r"(@\[A-Za-z0-9]+)|([^0-9A-Za-z \n])|(\w+:\/\/\S+)|^rt|www.+?", " ", text)

        expanded_text = []
        for word in text.split():
            expanded_text.append(word) 
        return expanded_text

def construct_vocab(text):
        vocab = {}

        vocab["sos"] = 0
        vocab["eos"] = 1
        vocab["unk"] = 2
        vocab["<pad>"] = 3
        
        for word in text:
            if word not in vocab:
                vocab[word] = len(vocab) # add a new type to the vocab
        return vocab

def process_data(test_data, vocab):
    process_data = test_data.copy()
    for  i, word in enumerate(test_data):
        if word not in vocab:
            process_data[i] = 'unk' 
    return process_data

def tokenize(text: str, tokenizer):
    return [tok.text.lower() for tok in tokenizer.tokenizer(text)]

def build_vocabulary(train, val, test, spacy_eng, spacy_fr, min_freq: int = 2, index: int = 0):
  def tokenize_eng(text: str):
    return tokenize(text, spacy_eng)    #returns list of strings

  def tokenize_fr(text: str):
    return tokenize(text, spacy_fr)    #returns list of strings
  
  # generate source vocabulary (only from train sentences)
  vocab = build_vocab_from_iterator(
        yield_tokens(train, tokenize_eng if index == 0 else tokenize_fr), # tokens for each english sentence (index 0)
        # yield_tokens(train + val + test, tokenize_eng if index == 0 else tokenize_fr), # tokens for each english sentence (index 0)
        min_freq=min_freq,     #retain only those words in vocab with freq atleast 2
        specials=["<sos>", "<eos>", "<pad>", "<unk>"],
  )

  # set default token for out-of-vocabulary words (OOV)
  vocab.set_default_index(vocab["<unk>"])   # vocab["David"] will be 3 as David has 'D' and will be considered OOV
  return vocab

class Embeddings(nn.Module):
  def __init__(self, vocab_size: int, d_model: int):
    """
    Args:
      vocab_size:     size of vocabulary
      d_model:        dimension of embeddings
    """
    # inherit from nn.Module
    super().__init__()   
     
    # embedding look-up table (lut)                          
    self.lut = nn.Embedding(vocab_size, d_model)   

    # dimension of embeddings 
    self.d_model = d_model                          

  def forward(self, x: Tensor):
    """
    Args:
      x:              input Tensor (batch_size, seq_length)
      
    Returns:
                      embedding vector
    """
    # embeddings by constant sqrt(d_model)
    return self.lut(x) * math.sqrt(self.d_model)
  
class PositionalEncoding(nn.Module):
  def __init__(self, d_model: int, dropout: float = 0.1, max_length: int = 5000):
    """
    Args:
      d_model:      dimension of embeddings
      dropout:      randomly zeroes-out some of the input
      max_length:   max sequence length
    """
    # inherit from Module
    super().__init__()     

    # initialize dropout                  
    self.dropout = nn.Dropout(p=dropout)      

    # create tensor of 0s
    pe = torch.zeros(max_length, d_model)    

    # create position column   
    k = torch.arange(0, max_length).unsqueeze(1)  

    # calc divisor for positional encoding 
    div_term = torch.exp(                                 
            torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model)
    )

    # calc sine on even indices
    pe[:, 0::2] = torch.sin(k * div_term)    

    # calc cosine on odd indices   
    pe[:, 1::2] = torch.cos(k * div_term)  

    # add dimension     
    pe = pe.unsqueeze(0)          

    # buffers are saved in state_dict but not trained by the optimizer                        
    self.register_buffer("pe", pe)                        

  def forward(self, x: Tensor):
    """
    Args:
      x:        embeddings (batch_size, seq_length, d_model)
    
    Returns:
                embeddings + positional encodings (batch_size, seq_length, d_model)
    """
    # add positional encoding to the embeddings
    x = x + self.pe[:, : x.size(1)].requires_grad_(False) 

    # perform dropout
    return self.dropout(x)
  
class MultiHeadAttention(nn.Module):
  def __init__(self, d_model: int = 512, n_heads: int = 8, dropout: float = 0.1):
    """
    Args:
        d_model:      dimension of embeddings
        n_heads:      number of self attention heads
        dropout:      probability of dropout occurring
    """
    super().__init__()
    assert d_model % n_heads == 0            # ensure an even num of heads
    self.d_model = d_model                   # 512 dim
    self.n_heads = n_heads                   # 8 heads
    self.d_key = d_model // n_heads          # assume d_value equals d_key | 512/8=64

    self.Wq = nn.Linear(d_model, d_model)    # query weights
    self.Wk = nn.Linear(d_model, d_model)    # key weights
    self.Wv = nn.Linear(d_model, d_model)    # value weights
    self.Wo = nn.Linear(d_model, d_model)    # output weights

    self.dropout = nn.Dropout(p=dropout)     # initialize dropout layer  

  def forward(self, query: Tensor, key: Tensor, value: Tensor, mask: Tensor = None):
    """
    Args:
       query:         query vector         (batch_size, q_length, d_model)
       key:           key vector           (batch_size, k_length, d_model)
       value:         value vector         (batch_size, s_length, d_model)
       mask:          mask for decoder     

    Returns:
       output:        attention values     (batch_size, q_length, d_model)
       attn_probs:    softmax scores       (batch_size, n_heads, q_length, k_length)
    """
    batch_size = key.size(0)                  
        
    # calculate query, key, and value tensors
    Q = self.Wq(query)                       # (64, 100, 256) x (256, 256) = (64, 100, 256)
    K = self.Wk(key)                         # (64, 100, 256) x (256, 256) = (64, 100, 256)
    V = self.Wv(value)                       # (64, 100, 256) x (256, 256) = (64, 100, 256)

    # split each tensor into n-heads to compute attention

    # query tensor
    Q = Q.view(batch_size,                   # (64, 100, 256) -> (64, 100, 8, 32) 
               -1,                           # -1 = q_length
               self.n_heads,              
               self.d_key  #32
               ).permute(0, 2, 1, 3)         # (64, 100, 8, 32)  -> (64, 8, 100, 32) = (batch_size, n_heads, q_length, d_key)
    # key tensor
    K = K.view(batch_size,                   
               -1,                           
               self.n_heads,              
               self.d_key
               ).permute(0, 2, 1, 3)         # (64, 100, 8, 32)  -> (64, 8, 100, 32) = (batch_size, n_heads, k_length, d_key)
    # value tensor
    V = V.view(batch_size,                   
               -1,                           
               self.n_heads, 
               self.d_key
               ).permute(0, 2, 1, 3)         # (64, 100, 8, 32)  -> (64, 8, 100, 32) = (batch_size, n_heads, v_length, d_key)
       
    # computes attention
    # scaled dot product -> QK^{T}
    scaled_dot_prod = torch.matmul(Q,        # (64, 8, 100, 32) x (64, 8, 32, 100) -> (64, 8, 100, 100) = (batch_size, n_heads, q_length, k_length)
                                   K.permute(0, 1, 3, 2)
                                   ) / math.sqrt(self.d_key)      # sqrt(64)
        
    # fill those positions of product as (-1e10) where mask positions are 0
    if mask is not None:   #shape: (64, 1, 99, 99) for masked attention  and (64, 1, 1, 100) for self and cross attention which is unmasked (just for padding)
      scaled_dot_prod = scaled_dot_prod.masked_fill(mask == 0, -1e10)

    # apply softmax 
    attn_probs = torch.softmax(scaled_dot_prod, dim=-1)
        
    # multiply by values to get attention
    A = torch.matmul(self.dropout(attn_probs), V)       # (64, 8, 100, 100) x (64, 8, 100, 32) -> (64, 8, 100, 64)
                                                        # (batch_size, n_heads, q_length, k_length) x (batch_size, n_heads, v_length, d_key) -> (batch_size, n_heads, q_length, d_key)

    # reshape attention back to (32, 10, 512)
    A = A.permute(0, 2, 1, 3).contiguous()              # (64, 8, 100, 32) -> (64, 100, 8, 32)
    A = A.view(batch_size, -1, self.n_heads*self.d_key) # (64, 100, 8, 32) -> (64, 100, 8*32) -> (64, 100, 256) = (batch_size, q_length, d_model)
        
    # push through the final weight layer
    output = self.Wo(A)                                 # (64, 100, 256) x (256, 256) = (64, 100, 256) 

    return output, attn_probs                           # return attn_probs for visualization of the scores
  
class PositionwiseFeedForward(nn.Module):
  def __init__(self, d_model: int, d_ffn: int, dropout: float = 0.1):
    """
    Args:
        d_model:      dimension of embeddings
        d_ffn:        dimension of feed-forward network
        dropout:      probability of dropout occurring
    """
    super().__init__()

    self.w_1 = nn.Linear(d_model, d_ffn)
    self.w_2 = nn.Linear(d_ffn, d_model)
    self.dropout = nn.Dropout(dropout)

  def forward(self, x):
    """
    Args:
        x:            output from attention (batch_size, seq_length, d_model)
       
    Returns:
        expanded-and-contracted representation (batch_size, seq_length, d_model)
    """
    # w_1(x).relu(): (batch_size, seq_length, d_model) x (d_model,d_ffn) -> (batch_size, seq_length, d_ffn)
    # w_2(w_1(x).relu()): (batch_size, seq_length, d_ffn) x (d_ffn, d_model) -> (batch_size, seq_length, d_model) 
    return self.w_2(self.dropout(self.w_1(x).relu()))
  
class EncoderLayer(nn.Module):  
  def __init__(self, d_model: int, n_heads: int, d_ffn: int, dropout: float):
    """
    Args:
        d_model:      dimension of embeddings
        n_heads:      number of heads
        d_ffn:        dimension of feed-forward network
        dropout:      probability of dropout occurring
    """
    super().__init__()
    # multi-head attention sublayer
    self.attention = MultiHeadAttention(d_model, n_heads, dropout)
    # layer norm for multi-head attention
    self.attn_layer_norm = nn.LayerNorm(d_model)

    # position-wise feed-forward network
    self.positionwise_ffn = PositionwiseFeedForward(d_model, d_ffn, dropout)
    # layer norm for position-wise ffn
    self.ffn_layer_norm = nn.LayerNorm(d_model)

    self.dropout = nn.Dropout(dropout)

  def forward(self, src: Tensor, src_mask: Tensor):
    """
    Args:
        src:          positionally embedded sequences   (batch_size, seq_length, d_model=256 here)
        src_mask:     mask for the sequences            (batch_size, 1, 1, seq_length)
    Returns:
        src:          sequences after self-attention    (batch_size, seq_length, d_model)
    """
    # pass embeddings through multi-head attention
    _src, attn_probs = self.attention(src, src, src, src_mask)

    # residual add and norm
    src = self.attn_layer_norm(src + self.dropout(_src))   #(b_size, 100, 256)
    
    # position-wise feed-forward network
    _src = self.positionwise_ffn(src)    #(b_size, 100, 256)

    # residual add and norm
    src = self.ffn_layer_norm(src + self.dropout(_src))    #(b_size, 100, 256)

    return src, attn_probs

class Encoder(nn.Module):
  def __init__(self, d_model: int, n_layers: int, 
               n_heads: int, d_ffn: int, dropout: float = 0.1):
    """
    Args:
        d_model:      dimension of embeddings
        n_layers:     number of encoder layers
        n_heads:      number of heads
        d_ffn:        dimension of feed-forward network
        dropout:      probability of dropout occurring
    """
    super().__init__()
    
    # create n_layers encoders 
    self.layers = nn.ModuleList([EncoderLayer(d_model, n_heads, d_ffn, dropout)
                                 for layer in range(n_layers)])

    self.dropout = nn.Dropout(dropout)
    
  def forward(self, src: Tensor, src_mask: Tensor):
    """
    Args:
        src:          embedded sequences                (batch_size, seq_length, d_model)
        src_mask:     mask for the sequences            (batch_size, 1, 1, seq_length)

    Returns:
        src:          sequences after self-attention    (batch_size, seq_length, d_model)
    """

    # pass the sequences through each encoder
    for layer in self.layers:
      src, attn_probs = layer(src, src_mask)

    self.attn_probs = attn_probs

    return src    ##(b_size, 100, 256)
  
class DecoderLayer(nn.Module):

  def __init__(self, d_model: int, n_heads: int, d_ffn: int, dropout: float):
    """
    Args:
        d_model:      dimension of embeddings
        n_heads:      number of heads
        d_ffn:        dimension of feed-forward network
        dropout:      probability of dropout occurring
    """
    super().__init__()
    # masked multi-head attention sublayer
    self.masked_attention = MultiHeadAttention(d_model, n_heads, dropout)
    # layer norm for masked multi-head attention
    self.masked_attn_layer_norm = nn.LayerNorm(d_model)

    # multi-head attention sublayer
    self.attention = MultiHeadAttention(d_model, n_heads, dropout)
    # layer norm for multi-head attention
    self.attn_layer_norm = nn.LayerNorm(d_model)
    
    # position-wise feed-forward network
    self.positionwise_ffn = PositionwiseFeedForward(d_model, d_ffn, dropout)
    # layer norm for position-wise ffn
    self.ffn_layer_norm = nn.LayerNorm(d_model)

    self.dropout = nn.Dropout(dropout)

  def forward(self, trg: Tensor, src: Tensor, trg_mask: Tensor, src_mask: Tensor):
    """
    Args:
        trg:          embedded sequences                (batch_size, trg_seq_length=99, d_model=256)
        src:          embedded sequences                (batch_size, src_seq_length=100, d_model)
        trg_mask:     mask for the sequences            (batch_size, 1, trg_seq_length, trg_seq_length)
        src_mask:     mask for the sequences            (batch_size, 1, 1, src_seq_length)

    Returns:
        trg:          sequences after self-attention    (batch_size, trg_seq_length, d_model)
        attn_probs:   self-attention softmax scores     (batch_size, n_heads, trg_seq_length, src_seq_length)
    """
    # pass trg embeddings through masked multi-head attention
    _trg, attn_probs = self.masked_attention(trg, trg, trg, trg_mask)

    # residual add and norm
    trg = self.masked_attn_layer_norm(trg + self.dropout(_trg))
    
    # pass trg and src embeddings through multi-head attention  #this is cross attention where Q comes from trg sentence and K, V comes from src sentence (no masking here)
    _trg, attn_probs = self.attention(trg, src, src, src_mask)

    # residual add and norm
    trg = self.attn_layer_norm(trg + self.dropout(_trg))

    # position-wise feed-forward network
    _trg = self.positionwise_ffn(trg)

    # residual add and norm
    trg = self.ffn_layer_norm(trg + self.dropout(_trg)) 

    return trg, attn_probs

class Decoder(nn.Module):
  def __init__(self, vocab_size: int, d_model: int, n_layers: int, 
               n_heads: int, d_ffn: int, dropout: float = 0.1):
    """
    Args:
        vocab_size:   size of the target vocabulary
        d_model:      dimension of embeddings
        n_layers:     number of encoder layers
        n_heads:      number of heads
        d_ffn:        dimension of feed-forward network
        dropout:      probability of dropout occurring
    """
    super().__init__()

    # create n_layers encoders 
    self.layers = nn.ModuleList([DecoderLayer(d_model, n_heads, d_ffn, dropout)
                                 for layer in range(n_layers)])
    
    self.dropout = nn.Dropout(dropout)

    # set output layer
    self.Wo = nn.Linear(d_model, vocab_size)
    
  def forward(self, trg: Tensor, src: Tensor, trg_mask: Tensor, src_mask: Tensor):
    """
    Args:
        trg:          embedded sequences                (batch_size, trg_seq_length, d_model)
        src:          encoded sequences from encoder    (batch_size, src_seq_length, d_model)
        trg_mask:     mask for the sequences            (batch_size, 1, trg_seq_length, trg_seq_length)
        src_mask:     mask for the sequences            (batch_size, 1, 1, src_seq_length)

    Returns:
        output:       sequences after decoder           (batch_size, trg_seq_length, vocab_size)
        attn_probs:   self-attention softmax scores     (batch_size, n_heads, trg_seq_length, src_seq_length)
    """

    # pass the sequences through each decoder
    for layer in self.layers:
      trg, attn_probs = layer(trg, src, trg_mask, src_mask)

    self.attn_probs = attn_probs

    return self.Wo(trg)
  
class Transformer(nn.Module):
  def __init__(self, encoder: Encoder, decoder: Decoder,
               src_embed: Embeddings, trg_embed: Embeddings,
               src_pad_idx: int, trg_pad_idx: int, device):
    """
    Args:
        encoder:      encoder stack                    
        decoder:      decoder stack
        src_embed:    source embeddings and encodings
        trg_embed:    target embeddings and encodings
        src_pad_idx:  padding index          
        trg_pad_idx:  padding index
        device:       cuda or cpu
    
    Returns:
        output:       sequences after decoder           (batch_size, trg_seq_length, vocab_size)
    """
    super().__init__()

    self.encoder = encoder
    self.decoder = decoder
    self.src_embed = src_embed  #nn.Sequential(src_embed, pos_enc)
    self.trg_embed = trg_embed   #nn.Sequential(src_embed, pos_enc)
    self.device = device
    self.src_pad_idx = src_pad_idx
    self.trg_pad_idx = trg_pad_idx
    
  def make_src_mask(self, src: Tensor):
    # assign 1 to tokens that need attended to and 0 to padding tokens, then add 2 dimensions
    src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)

    return src_mask

  def make_trg_mask(self, trg: Tensor):

    seq_length = trg.shape[1]  #99

    # assign True to tokens that need attended to and False to padding tokens, then add 2 dimensions (padding token, we have this in encoder i.e src mask as well)
    trg_mask = (trg != self.trg_pad_idx).unsqueeze(1).unsqueeze(2) # (batch_size, 1, 1, seq_length)

    # generate subsequent mask (lookahead mask)
    trg_sub_mask = torch.tril(torch.ones((seq_length, seq_length), device=self.device)).bool() # (batch_size, 1, seq_length, seq_length)

    # bitwise "and" operator | 0 & 0 = 0, 1 & 1 = 1, 1 & 0 = 0
    trg_mask = trg_mask & trg_sub_mask

    return trg_mask

  def forward(self, src: Tensor, trg: Tensor):   #src: batch of source lang sentences,  trg: batch of target lang sentences
    # src shape: (b_size, 100) , trg_shape: (b_size, 99) where src_seq_len=99, trg_seq_len=99

    # create source and target masks     
    src_mask = self.make_src_mask(src) # (batch_size, 1, 1, src_seq_length)
    trg_mask = self.make_trg_mask(trg) # (batch_size, 1, trg_seq_length, trg_seq_length)

    # push the src through the encoder layers
    src = self.encoder(self.src_embed(src), src_mask)  # (batch_size, src_seq_length, d_model)

    # decoder output and attention probabilities
    output = self.decoder(self.trg_embed(trg), src, trg_mask, src_mask)

    return output   #(b_size, trg_seq_len, trg_vocab_size)
  
def get_transformer_model(device, src_vocab, trg_vocab, n_layers=2, n_heads=8, model_dim=256,
                                   feed_forward_dim=512, max_seq_length=100, dropout = 0.1):
  
  encoder = Encoder(model_dim, n_layers, n_heads, feed_forward_dim, dropout)

  decoder = Decoder(len(trg_vocab), model_dim, n_layers, n_heads, feed_forward_dim, dropout)
    
  src_embed = Embeddings(len(src_vocab), model_dim)   #source embedding matrix  (src_vocab_len, model_dim) - this is a look up table (embedding for each token in src corpus)
  
  trg_embed = Embeddings(len(trg_vocab), model_dim)  #target embedding matrix
  
  pos_enc = PositionalEncoding(model_dim, dropout, max_seq_length)    #positional encoding matrix

  model = Transformer(encoder, decoder, nn.Sequential(src_embed, pos_enc), 
                      nn.Sequential(trg_embed, pos_enc),
                      src_pad_idx=src_vocab.get_stoi()["<pad>"], 
                      trg_pad_idx=trg_vocab.get_stoi()["<pad>"],
                      device=device)

  # initialize parameters with Xavier/Glorot
  for p in model.parameters():
    if p.dim() > 1:
      nn.init.xavier_uniform_(p)

  return model

def train_network(args, train_dataset, val_dataset, test_dataset):
    training_start_time = time.time()

    LM_model = get_transformer_model(device, vocab_src, vocab_trg,
                                   n_layers=2, n_heads=8, model_dim=256,
                                   feed_forward_dim=512, max_seq_length=100)
    
    LM_model = LM_model.to(device)

    #-----------------------------------------------------------------------------#
    #--------------------------------- main loop ---------------------------------#
    #-----------------------------------------------------------------------------#
    is_LSTM = True
    trainer = Trainer_transformer_jayaram(LM_model, train_dataset, val_dataset, is_LSTM, device)

    for i in range(args.epochs):
        print(f'Epoch {i} / {args.epochs} | {args.save_ckpt}')
        trainer.train_Transformer(device, i, n_train_steps=args.number_of_steps_per_epoch)

    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    print("")
    print("Done.")
    print("")
    print("Total training time: {} seconds.".format(time.time() - training_start_time))
    print("")

if __name__ == "__main__":

    # Parse input arguments
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "-o",
        "--output-dir",
        help="Path to output directory for training results. Nothing specified means training results will NOT be saved.",
    )

    parser.add_argument(
        "-save_ckpt",
        "--save_ckpt",
        type=str,
        default="/home/jayaram/research/research_tracks/table_top_rearragement/test_diffusion_planning/logs/maze2d-large-v1/diffusion",
        help="save checkpoints of diffusion model while training",
    )

    parser.add_argument(
        "-e", "--epochs", type=int, default=30, help="Number of epochs to train."
    )

    parser.add_argument(
        "-n_steps_per_epoch",
        "--number_of_steps_per_epoch",
        type=int,
        default=1000,
        help="Number of steps per epoch",
    )

    # parser.add_argument(
    #     "-b",
    #     "--batch-size",
    #     type=int,
    #     required=True,
    #     help="The number of samples per batch used for training.",
    # )

    parser.add_argument(
        "-lr",
        "--learning-rate",
        type=float,
        default=0.001,
        help="The learning rate used for the optimizer.",
    )

    parser.add_argument(
        "-w",
        "--num-workers",
        type=int,
        default=8,
        help='The number of subprocesses ("workers") used for loading the training data. 0 means that no subprocesses are used.',
    )
    args = parser.parse_args()

# train the Transformer model
# train_network(args, train_dataset, val_dataset, vocab, word_emd_model)

# max_len = 0
# for input in inputs: 
#     words = input.split(" ")
#     max_len = max(max_len, len(words))


#######################################################################################################################################
# Load the English spaCy model
spacy_eng = spacy.load("en_core_web_sm")
# Load the French spaCy model
spacy_fr = spacy.load("fr_core_news_sm")

# Example text in English and French
text_eng = "This is an example sentence in English."
text_fr = "Ceci est une phrase exemple en français."

# # Process text with the English model
# doc_eng = nlp_eng(text_eng)
# # Process text with the French model
# doc_fr = nlp_fr(text_fr)
# # Accessing tokens and their attributes
# for token in doc_eng:
#     print(token.text, token.pos_, token.dep_)
# for token in doc_fr:
#     print(token.text, token.pos_, token.dep_)

# Define paths to your English and French text files for train, validation, and test
train_en_path = "ted-talks-corpus/train.en"
train_fr_path = "ted-talks-corpus/train.fr"
val_en_path = "ted-talks-corpus/dev.en"
val_fr_path = "ted-talks-corpus/dev.fr"
test_en_path = "ted-talks-corpus/test.en"
test_fr_path = "ted-talks-corpus/test.fr"

# Load train, validation, and test data pipelines for English and French
train_en = load_text_data(train_en_path)  #30k sentences
train_fr = load_text_data(train_fr_path)
val_en = load_text_data(val_en_path)    #887 sentences
val_fr = load_text_data(val_fr_path)
test_en = load_text_data(test_en_path)  #1305 sentences
test_fr = load_text_data(test_fr_path)

if not os.path.exists("vocab.pt"):
    vocab_src = build_vocabulary(train_en, val_en, test_en, spacy_eng, spacy_fr, index=0)    #build vocab for eng lang (give index to each word in corpus)
    vocab_trg = build_vocabulary(train_fr, val_fr, test_fr, spacy_eng, spacy_fr, index=1)     #build vocab for french lang

    torch.save((vocab_src, vocab_trg), "vocab.pt")
else:
    # load the vocab if it exists
    vocab_src, vocab_trg = torch.load("vocab.pt")

print("Finished.\nVocabulary sizes:")
print("\tSource:", len(vocab_src))
print("\tTarget:", len(vocab_trg))

# Vocabulary sizes: 
#         Source: 13810 (eng)
#         Target: 16994  (fr)

# processed data
train_data_en = data_process(train_en, spacy_eng, index = 0)  #list of tensors where each tensor has indices of those words in vocab
val_data_en = data_process(val_en, spacy_eng, index = 0)
test_data_en = data_process(test_en, spacy_eng, index = 0)
train_data_fr = data_process(train_fr, spacy_fr, index = 1)
val_data_fr = data_process(val_fr, spacy_fr, index = 1)
test_data_fr = data_process(test_fr, spacy_fr, index = 1)

SOS_IDX = vocab_trg['<sos>']    #0
EOS_IDX = vocab_trg['<eos>']    #1
PAD_IDX = vocab_trg['<pad>']    #2

def generate_batch(data_batch):

  eng_batch, fr_batch = [], []

  # for each sentence
  for (eng_item, fr_item) in data_batch:
    # add <bos> and <eos> indices before and after the sentence
    eng_temp = torch.cat([torch.tensor([SOS_IDX]), eng_item, torch.tensor([EOS_IDX])], dim=0).to(device)
    fr_temp = torch.cat([torch.tensor([SOS_IDX]), fr_item, torch.tensor([EOS_IDX])], dim=0).to(device)

    # add padding
    eng_batch.append(pad(eng_temp,(0, # dimension to pad
                            MAX_PADDING - len(eng_temp), # amount of padding to add
                          ),value=PAD_IDX,))
    
    # add padding
    fr_batch.append(pad(fr_temp,(0, # dimension to pad
                            MAX_PADDING - len(fr_temp), # amount of padding to add
                          ),
                          value=PAD_IDX,))
    
  return torch.stack(eng_batch), torch.stack(fr_batch)

MAX_PADDING = 100
BATCH_SIZE = 128

# Combine English and French data for train, val, and test
train_data = list(zip(train_data_en, train_data_fr))
val_data = list(zip(val_data_en, val_data_fr))
test_data = list(zip(test_data_en, test_data_fr))

eng_sentences_below_50_words = 0
fr_sentences_below_50_words = 0
eng_sentences_below_100_words = 0
fr_sentences_below_100_words = 0
eng_sentences_below_500_words = 0
fr_sentences_below_500_words = 0

max_eng_len = 0
max_eng_sent_idx = 0
max_fr_len = 0
max_fr_sent_idx = 0
for i, (eng_item, fr_item) in enumerate(train_data):
    if len(eng_item) <= 50:
        eng_sentences_below_50_words += 1   #28715
    if len(fr_item) <= 50:
        fr_sentences_below_50_words += 1    #28321

    if len(eng_item) <= 100:
        eng_sentences_below_100_words += 1    #29949 (use MAX_PADDING  as 100 -- meaning including <pad> tokens, sentence should have 100 tokens)
    if len(fr_item) <= 100:
        fr_sentences_below_100_words += 1     #29911

    if len(eng_item) <= 500:
        eng_sentences_below_500_words += 1     #29998
    if len(fr_item) <= 500:
        fr_sentences_below_500_words += 1      #29998

    if len(eng_item) > max_eng_len:
        max_eng_len = len(eng_item)
        max_eng_sent_idx = i    #20214 sentence has max len : train_en[20214]
    if len(fr_item) > max_fr_len:
        max_fr_len = len(fr_item)
        max_fr_sent_idx = i

train_loader = DataLoader(to_map_style_dataset(train_data), batch_size=BATCH_SIZE,
                        shuffle=True, drop_last=True, collate_fn=generate_batch)
# train_iter = iter(train_iter)
# batch = next(train_iter)
# batch[0].shape and batch[1].shape: torch.Size([128, 100])

valid_loader = DataLoader(to_map_style_dataset(val_data), batch_size=BATCH_SIZE,
                        shuffle=True, drop_last=True, collate_fn=generate_batch)

test_loader = DataLoader(to_map_style_dataset(test_data), batch_size=BATCH_SIZE,
                       shuffle=True, drop_last=True, collate_fn=generate_batch)


# train the Transformer model
train_network(args, train_data, val_data, test_data)