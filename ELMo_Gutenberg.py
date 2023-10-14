import os
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import numpy as np
from matplotlib import pyplot as plt
import random
import argparse
import time

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
import pandas as pd
import json
import torchtext

# from torchtext.datasets import IMDB
from torchtext.data import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
# from torch.nn.utils.rnn import pack_padded_sequence
from torchtext.vocab import vocab
import itertools
# from torchtext.datasets import TextClassificationDataset
from torchtext.vocab import build_vocab_from_iterator
# from torchtext.functional import sequential_transforms
from torchtext.data.functional import to_map_style_dataset

random.seed(32)

from Trainer import Trainer_jayaram

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# wandb setup
number = 1
NAME = "model" + str(number)
ID = 'LSTM_training_' + str(number)
run = wandb.init(project='LSTM_training', name = NAME, id = ID)



class GloveWordEmbeddings:
    def __init__(self, glove_file_path, word2vec_output_file):
        self.glove_file_path = glove_file_path
        self.word2vec_output_file = word2vec_output_file

    def convert_to_word2vec_format(self):
        # Convert GloVe to Word2Vec format
        glove2word2vec(self.glove_file_path, self.word2vec_output_file)

    def load_word2vec_model(self):
        # Load Word2Vec model
        model = KeyedVectors.load_word2vec_format(self.word2vec_output_file, binary=False)
        return model
    
# Define a custom dataset class
class AGNewsDataset(Dataset):
    def __init__(self, csv_file, vocab_file, train = True, max_sequence_length=128):
        self.data = pd.read_csv(csv_file, header=None, names=['Class Index', 'Description'])   #120k rows
        self.data['Description'] = self.data['Description'] + " ."
        # self.data['Description'].head()

        if(train):
            if not os.path.exists("AG_data/train"):
                os.makedirs("AG_data/train")
            
                for i in range(0, self.data.shape[0],6):
                    text = "\n".join(self.data['Description'][i:i+6].tolist())
                    fp = open("AG_data/train/"+str(i)+".txt","w")
                    fp.write(text)
                    fp.close()
        else:
            if not os.path.exists("AG_data/val"):
                os.makedirs("AG_data/val")
            
                for i in range(0, self.data.shape[0],6):
                    text = "\n".join(self.data['Description'][i:i+6].tolist())
                    fp = open("AG_data/val/"+str(i)+".txt","w")
                    fp.write(text)
                    fp.close()

        self.vocab = self.load_vocab(vocab_file)
        self.vectorizer = CountVectorizer(
            vocabulary=self.vocab,
            max_features=max_sequence_length,
            stop_words=None,
            max_df=0.95,
            min_df=2
        )

        self.X = self.vectorizer.fit_transform(self.data['Class Index'] + ' ' + self.data['Description']).toarray()
        self.y = self.data['Class Index'] - 1  # Adjust labels to start from 0

    def load_vocab(self, vocab_file):
        with open(vocab_file, 'r') as f:
            vocab = [line.strip() for line in f]
        return vocab

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = {
            'text': torch.tensor(self.X[idx], dtype=torch.float32),
            'label': torch.tensor(self.y[idx], dtype=torch.int64)
        }
        return sample

class CharConv(nn.Module):
    
    def __init__(self):
        super(CharConv, self).__init__()
        
        # Embedding layer
        self.char_embedding = nn.Embedding(CHAR_VOCAB_SIZE, CHAR_EMBED_DIM)
        
        # Conv layers
        self.conv1 = nn.Conv2d(CHAR_EMBED_DIM, 2, 1)
        self.conv2 = nn.Conv2d(CHAR_EMBED_DIM, 2, (1, 2))
        self.conv3 = nn.Conv2d(CHAR_EMBED_DIM, 4, (1, 3))
        self.conv4 = nn.Conv2d(CHAR_EMBED_DIM, 8, (1, 4))
        self.conv5 = nn.Conv2d(CHAR_EMBED_DIM, 16, (1, 5))
        self.conv6 = nn.Conv2d(CHAR_EMBED_DIM, 32, (1, 6))
        self.conv7 = nn.Conv2d(CHAR_EMBED_DIM, 64, (1, 7))
        self.convs = [
            self.conv1, self.conv2, 
            self.conv3, self.conv4, 
            self.conv5, self.conv6, 
            self.conv7,
        ]
        
    
    def forward(self, x):
        # character-level convolution
        x = self.char_embedding(x).permute(0,3,1,2)
        x = [conv(x) for conv in self.convs]
        x = [F.max_pool2d(x_c, kernel_size=(1, x_c.shape[3])) for x_c in x]
        x = [torch.squeeze(x_p, dim=3) for x_p in x]
        x = torch.hstack(x)  # 1, n_batch, concat_length
        
        return x

class BiLSTM(nn.Module):       # Bi-LSTM
    def __init__(self):
        super(BiLSTM, self).__init__()
        # two bidirectional LSTM layers (lstm_f1 and lstm_r1) with input size 128 and hidden size 128.
        self.lstm_forward = nn.LSTM(128, 128)
        self.lstm_reverse = nn.LSTM(128, 128)
        self.dropout = nn.Dropout(0.1)  # to prevent overfitting

        self.proj = nn.Linear(128, 64, bias=False)  #for dim reduction

        # 2 more LSTM layers for fwd and rev direction (lstm_f2 and lstm_r2) follow with input size 64 and hidden size 128 
        self.lstm_forward2 = nn.LSTM(64, 128)
        self.lstm_reverse2 = nn.LSTM(64, 128)
    
    def forward(self, x):
        ## input shape:
        # seq_len, batch_size, 128
        
        # 1st LSTM layer
        x_fwd = x  #(seq_len, batch_size, 128).
        x_rev = x.flip(dims=[0])  #elements of the tensor x are reversed along the sequence dimension

            # generic LSTM :
            # input shape => [seq_len, batch_size, embedding_dim]
            # output shape => [seq_len, batch_size, hidden_size]
            # hidden shape => [num_layers, batch_size, hidden_size]
            # cell shape => [num_layers, batch_size, hidden_size]

        # Forward pass
        out_fwd1, (hidden_fwd1, __) = self.lstm_forward(x_fwd)    #hidden_fwd1 : (1, batch_size, hidden_size)
        out_fwd1 = self.dropout(out_fwd1)
        # Backward pass
        out_rev1, (hidden_rev1, __) = self.lstm_reverse(x_rev)    #hidden_rev1 : (1, batch_size, hidden_size)
        out_rev1 = self.dropout(out_rev1)
        hidden_1 = torch.stack((hidden_fwd1, hidden_rev1)).squeeze(dim=1)

        # main + skip connection : Skip connections are added between the input x and the output of the first LSTM layer after projection (x2_fwd and x2_rev).
        x_fwd2 = self.projection(out_fwd1 + x_fwd)
        x_rev2 = self.projection(out_rev1 + x_rev)
        
        # 2nd LSTM layer
        _, (hidden_fwd2, __) = self.lstm_forward2(x_fwd2)
        _, (hidden_rev2, __) = self.lstm_reverse2(x_rev2)
        hidden_2 = torch.stack((hidden_fwd2, hidden_rev2)).squeeze(dim=1)
        
        return hidden_1, hidden_2
    
class ELMo(nn.Module):
    """
    Bidirectional language model (will be pretrained)
    1.) ELMo has forward and backward lang models each of which is trained thru a common Bi-LSTM.
    2.) Bi-LSTM is stacked (has 3 layers). First layer is layer of non-contexual word embeddings (EX: word2vec (or) character CNN - in original ELMo) and second, third layers are biLSTM layers
    3.) Forward lng model is trained to predict next word given prefix, back lang model is trained to predict prev word. Its a word prediction given context in either case.
    4.) Once wts are trained using forard, backward lang modelling obj, we can use wts for downstream task (as this pretrained model now has syntactic with lower layers and semantic knowledge of the lang with higher layers).
    5.) To get contexual word representation of a word, add the representations from all the layers including non-contexual word embeddings.
    5.) Finetune model for downstream task (5-way sentiment classification task for the assign).
    """
    def __init__(self, glove_embedding, bi_lstm):
        super(ELMo, self).__init__()
        
        # Highway connection
        self.highway = nn.Linear(128, 128)
        self.transform = nn.Linear(128, 128)
        
        # Word2Vec embeddings
        self.glove_embedding = glove_embedding
        
        # Bidirectional LSTM
        self.bi_lstm = bi_lstm
        
    def forward(self, x):
        # Glove embeddings (assuming x is a tensor with word indices)
        x = self.glove_embedding(x)
        
        # Highway
        h = self.highway(x)
        t_gate = torch.sigmoid(self.transform(x))
        c_gate = 1 - t_gate
        x_ = h * t_gate + x * c_gate
        
        # Bidirectional LSTM
        x1, x2 = self.bi_lstm(x_)
        
        return x1, x2
    
class ELMo_with_character_CNN(nn.Module):
    """
    Bidirectional language model (will be pretrained)
    1.) ELMo has forward and backward lang models each of which is trained thru a common Bi-LSTM.
    2.) Bi-LSTM is stacked (has 2 layers). First layer is layer of non-contexual word embeddings (EX: word2vec (or) character CNN - in original ELMo) and second layer
    3.) Forward lng model is trained to predict next word given prefix, back lang model is trained to predict prev word. Its a word prediction given context in either case.
    4.) Once wts are trained using forard, backward lang modelling obj, we can use wts for downstream task (as this pretrained model now has syntactic with lower layers and semantic knowledge of the lang with higher layers).
    5.) To get contexual word representation of a word, add the representations from all the layers including non-contexual word embeddings.
    5.) Finetune model for downstream task (5-way sentiment classification task for the assign).
    """
    def __init__(self, char_cnn, bi_lstm):
        super(ELMo, self).__init__()
        
        # Highway connection
        self.highway = nn.Linear(128, 128)   #just a linear layer
        self.transform = nn.Linear(128, 128)

        #two main components: a character-level CNN (char_cnn) and a bidirectional LSTM (bi_lstm).
        self.char_cnn = char_cnn
        self.bi_lstm = bi_lstm
        
    def forward(self, x):
        # Character-level convolution
        x = self.char_cnn(x)
        x = x.permute(2, 0, 1)   #x is permuted to have dimensions (seq_len, batch_size, 128).
        
        # Highway connections are applied to the character-level representations. These connections allow the model to control how much information is passed through and how much is transformed. This can help mitigate the vanishing gradient problem in deep networks.
        h = self.highway(x)
        t_gate = torch.sigmoid(self.transform(x))
        c_gate = 1 - t_gate
        x_ = h * t_gate + x * c_gate
        
        # Bi-LSTM
        x1, x2 = self.bi_lstm(x_)
        
        return x, x1, x2
    
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
        
def train_network(args, train_dataset, val_dataset, vocabulary):
    training_start_time = time.time()

    #load model architecture 
    vocab_size = len(vocabulary)  # Size of your vocabulary
    embedding_dim = 300  # Dimension of word embeddings
    hidden_dim = 512  # Dimension of hidden state in LSTM
    num_layers = 1  # Number of LSTM layers

    LM_model = ELMo(embedding_dim, hidden_dim, num_layers, vocab_size)
    LM_model = LM_model.to(device)

    #-----------------------------------------------------------------------------#
    #--------------------------------- main loop ---------------------------------#
    #-----------------------------------------------------------------------------#
    is_LSTM = True
    trainer = Trainer_jayaram(LM_model, train_dataset, val_dataset, is_LSTM)

    for i in range(args.epochs):
        print(f'Epoch {i} / {args.epochs} | {args.save_ckpt}')
        trainer.train_LSTM(device, i, n_train_steps=args.number_of_steps_per_epoch)

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
        "-w_sz", "--window_size", type=int, default=5, help="Number of words in the context."
    )

    parser.add_argument(
        "-e", "--epochs", type=int, default=2, help="Number of epochs to train."
    )

    parser.add_argument(
        "-n_steps_per_epoch",
        "--number_of_steps_per_epoch",
        type=int,
        default=100,
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



train_dataset, test_dataset  = torchtext.datasets.AG_NEWS()
target_classes = ["World", "Sports", "Business", "Sci/Tec"]
tokenizer = get_tokenizer("basic_english")

def build_vocab(datasets):
    for dataset in datasets:
        for _, text in dataset:
            yield tokenizer(text)

vocab = build_vocab_from_iterator(build_vocab([train_dataset, test_dataset]), specials=["<UNK>"])
vocab.set_default_index(vocab["<UNK>"])

vectorizer = CountVectorizer(vocabulary=vocab.get_itos(), tokenizer=tokenizer)

def vectorize_batch(batch):
    Y, X = list(zip(*batch))
    # X = vectorizer.transform(X).todense()
    return X, torch.tensor(Y) - 1 ## We have deducted 1 from target names to get them in range [0,1,2,3] from [1,2,3,4]

train_dataset, test_dataset = to_map_style_dataset(train_dataset), to_map_style_dataset(test_dataset)

train_loader = DataLoader(train_dataset, batch_size=256, collate_fn=vectorize_batch)
test_loader  = DataLoader(test_dataset, batch_size=256, collate_fn=vectorize_batch)

ngrams = 2
x = []
y = []

for sentences, class_labels in train_loader:
    for sentence in sentences:
        for i in range(len(sentence)-ngrams):
            text = sentence[i:i+ngrams]
            label = sentence[i+ngrams]
            x.append(text.tolist())
            y.append(label.tolist())

with open("./IMBD_bigram.json", "w") as w:
    json.dump({"data": x, "label": y}, w)



