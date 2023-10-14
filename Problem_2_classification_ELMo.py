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
import random
random.seed(32)
from torch.nn.utils.rnn import pad_sequence

from Trainer_classification import Trainer_classification_jayaram

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# wandb setup
number = 1
NAME = "model_dropout_02_" + str(number)
ID = 'ELMo_training_classification_' + str(number)
run = wandb.init(project='ELMo_training_classification_', name = NAME, id = ID)

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

class AGNews_Dataset(Dataset):
    def __init__(self, data, labels, window_size, vocab, word_emb_model, is_train = True, transform = None):
        self.is_train = is_train
        self.transform = transform
        self.window_size = window_size
        self.data = data    # list of sentences
        self.labels = labels    #list of class labels
        self.n_sentences = len(data)   #no of train sentences
        self.vocab = vocab
        self.word_emb_model = word_emb_model
        self.fixed_length = window_size

    def __len__(self):
        # return self.n_words - self.window_size
        return self.n_sentences

    def __getitem__(self, idx):
        sentence = self.data[idx]
        label = self.labels[idx]
        words = sentence.split(" ")
        words = [string for string in words if string]
        rev_words = words[::-1]

        # for forward LM        
        prefix_sentence = words  #list of prefix words
        if len(words) >= self.fixed_length:
            prefix_sentence = words[:self.fixed_length]  # Truncate if longer
            prefix_sentence.insert(0, 'sos')
            prefix_sentence.append('eos')
        else:
            num_padding = self.fixed_length - len(words) 
            prefix_sentence = words
            prefix_sentence.insert(0, 'sos')
            prefix_sentence.append('eos')
            prefix_sentence = prefix_sentence + ['<pad>'] * num_padding
 
        #get prefix
        prefix_word_embeddings = []
        # Define the unk token to represent out-of-vocabulary words
        unk_token = 'unk'
        # Iterate through the words in the window
        for word in prefix_sentence:
            if(word in self.word_emb_model.stoi) :
                # If the word is in the GloVe vocabulary, get its embedding
                word_embedding = self.word_emb_model.vectors[self.word_emb_model.stoi[word]]
            else:
                # If the word is not in the GloVe vocabulary, use the embedding of unk
                word_embedding = self.word_emb_model.vectors[self.word_emb_model.stoi[unk_token]]
            prefix_word_embeddings.append(word_embedding)

        # prefix_word_embeddings = [self.word_emb_model[word] for word in prefix_window]
        fwd_prefix_emb = np.concatenate(prefix_word_embeddings)
        fwd_prefix_emb = np.reshape(fwd_prefix_emb, (self.window_size +  2, -1))   #(seq_len, embed_dim for LSTM)

            # print(labels)     

        ###############################################################################################################
        # for reverse LM        
        rev_prefix_sentence = rev_words  #list of prefix words
        if len(rev_words) >= self.fixed_length:
            rev_prefix_sentence = rev_words[:self.fixed_length]  # Truncate if longer
            rev_prefix_sentence.insert(0, 'sos')
            rev_prefix_sentence.append('eos')
        else:
            num_padding = self.fixed_length - len(rev_words) 
            rev_prefix_sentence = rev_words
            rev_prefix_sentence.insert(0, 'sos')
            rev_prefix_sentence.append('eos')
            rev_prefix_sentence = rev_prefix_sentence + ['<pad>'] * num_padding

        #get prefix
        prefix_word_embeddings = []
        # Define the unk token to represent out-of-vocabulary words
        unk_token = 'unk'
        # Iterate through the words in the window
        for word in rev_prefix_sentence:
            if word in self.word_emb_model.stoi:
                # If the word is in the GloVe vocabulary, get its embedding
                word_embedding = self.word_emb_model.vectors[self.word_emb_model.stoi[word]]
            else:
                # If the word is not in the GloVe vocabulary, use the embedding of unk
                word_embedding = self.word_emb_model.vectors[self.word_emb_model.stoi[unk_token]]
            prefix_word_embeddings.append(word_embedding)

        # prefix_word_embeddings = [self.word_emb_model[word] for word in prefix_window]
        rev_prefix_emb = np.concatenate(prefix_word_embeddings)
        rev_prefix_emb = np.reshape(rev_prefix_emb, (self.window_size +  2, -1))   #(seq_len, embed_dim for LSTM)
        
        return ((fwd_prefix_emb, rev_prefix_emb, label))
        # return sentence

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
        self.lstm_forward = nn.LSTM(300, 300)
        self.lstm_reverse = nn.LSTM(300, 300)
        self.dropout = nn.Dropout(0.2)  # to prevent overfitting

        self.projection = nn.Linear(300, 128, bias=False)  #for dim reduction

        # 2 more LSTM layers for fwd and rev direction (lstm_f2 and lstm_r2) follow with input size 64 and hidden size 128 
        self.lstm_forward2 = nn.LSTM(128, 300)
        self.lstm_reverse2 = nn.LSTM(128, 300)
    
    def forward(self, x_fwd, x_rev):
        ## input shape:
        # seq_len, batch_size, 300

            # generic LSTM :
            # input shape => [seq_len, batch_size, embedding_dim]
            # output shape => [seq_len, batch_size, hidden_size]
            # hidden shape => [num_layers, batch_size, hidden_size]
            # cell shape => [num_layers, batch_size, hidden_size]

        # Forward pass
        out_fwd1, (hidden_fwd1, __) = self.lstm_forward(x_fwd)    #hidden_fwd1 : (1, batch_size = 256, hidden_size = 300)
        #out_fwd1 : (seq_len=52, batch_size = 256, hidden_size = 300)
        out_fwd1 = self.dropout(out_fwd1)

        # Backward pass (use reverse LSTM)
        out_rev1, (hidden_rev1, __) = self.lstm_reverse(x_rev)    #hidden_rev1 : (1, batch_size, hidden_size)
        out_rev1 = self.dropout(out_rev1)
        #out_rev1 : (seq_len=52, batch_size = 256, hidden_size = 300)
        hidden_1 = torch.stack((hidden_fwd1, hidden_rev1)).squeeze(dim=1)   #(2, 256, 300)

        # main + skip connection : Skip connections are added between the input x and the output of the first LSTM layer after projection (x2_fwd and x2_rev).
        x_fwd2 = self.projection(out_fwd1 + x_fwd)
        x_rev2 = self.projection(out_rev1 + x_rev)
        
        # 2nd LSTM layer
        out_fwd2, (hidden_fwd2, __) = self.lstm_forward2(x_fwd2)  #hidden_fwd1 : (1, batch_size = 256, hidden_size = 300)
        out_rev2, (hidden_rev2, __) = self.lstm_reverse2(x_rev2)   #hidden_rev1 : (1, batch_size = 256, hidden_size = 300)
        hidden_2 = torch.stack((hidden_fwd2, hidden_rev2)).squeeze(dim=1)  #(2, 256, 300)
        
        return ((out_fwd1, out_fwd2), (out_rev1, out_rev2), (hidden_1, hidden_2))
    
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
    def __init__(self, glove_embedding, bi_lstm, vocab_size):
        super(ELMo, self).__init__()
        
        # Highway connection
        self.highway = nn.Linear(300, 300)
        self.transform = nn.Linear(300, 300)
        
        # Word2Vec embeddings
        self.glove_embedding = glove_embedding
        
        # Bidirectional LSTM
        self.bi_lstm = bi_lstm
    
        self.W = nn.Linear(300, vocab_size)
        
    def forward(self, x_fwd, x_rev):
        # Glove embeddings (assuming x is a tensor with word indices)
        # x = self.glove_embedding(x)
        
        # Fwd LM
        # Highway
        h = self.highway(x_fwd)   #x: (52, 256, 300),  h: (52, 256, 300)
        t_gate = torch.sigmoid(self.transform(x_fwd))   # (52, 256, 300)
        c_gate = 1 - t_gate  
        x_fwd = h * t_gate + x_fwd * c_gate  # (52, 256, 300)

        # Rev LM
        # Highway
        h = self.highway(x_rev)   #x: (52, 256, 300),  h: (52, 256, 300)
        t_gate = torch.sigmoid(self.transform(x_rev))   # (52, 256, 300)
        c_gate = 1 - t_gate  
        x_rev = h * t_gate + x_rev * c_gate  # (52, 256, 300)
        
        # Bidirectional LSTM
        ((out_fwd1, out_fwd2), (out_rev1, out_rev2), (hidden_1, hidden_2)) = self.bi_lstm(x_fwd, x_rev)

        #map to vocab size
        logits_fwd2 = self.W(out_fwd2)   #(seq_len, batch_size, vocab_size) 
        logits_rev2 = self.W(out_rev2)   #(seq_len, batch_size, vocab_size) 
        
        return ((out_fwd1, out_fwd2), (out_rev1, out_rev2), (hidden_1, hidden_2))
    
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
    
class classification_LSTM(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, n_layers, elmo, num_classes = 4, bidirectional = True):
        super().__init__()
        # self.embedding = nn.Embedding(input_dim, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=n_layers, bidirectional=bidirectional)
        self.W = nn.Linear(hidden_dim * 2, num_classes)

        self.elmo = elmo
        # Freeze ELMo and set it to evaluation mode
        for param in self.elmo.parameters():
            param.requires_grad = False
        self.elmo.eval()

    def forward(self, x_fwd, x_rev):
        #Step1: Get elmo embeddings

        (out_fwd1, out_fwd2), (out_rev1, out_rev2), _ = self.elmo(x_fwd, x_rev)   #logits in this case (seq_len, B, vocab_size) 
        flipped_out_rev1 = torch.flip(out_rev1, dims=[0])
        flipped_out_rev2 = torch.flip(out_rev2, dims=[0])

        # Assuming you have weights for the summation
        # weight_fwd1 = 0.5
        # weight_fwd2 = 0.3
        # weight_x_fwd = 0.2

        weights = nn.Parameter(torch.rand(3)).to(device)
        weights_normalized = nn.functional.softmax(weights, dim=-1)

        # Concatenate the outputs
        concatenated_output_fwd1 = torch.cat((out_fwd1, flipped_out_rev1), dim=2)    #(seq_len, B, 300+300)
        concatenated_output_fwd2 = torch.cat((out_fwd2, flipped_out_rev2), dim=2)
        concatenated_input = torch.cat((x_fwd, x_fwd), dim=2)    #(seq_len, B, 300+300)

        # Apply weighted summation
        elmo_embedding = (
            weights_normalized[0] * concatenated_output_fwd1
            + weights_normalized[1] * concatenated_output_fwd2
            + weights_normalized[2] * concatenated_input
        )    #(seq_len, B, 300+300)


        # input shape => [seq_len, batch_size, embedding_dim]
        output, (hidden, cell) = self.lstm(elmo_embedding)   #([2, 64, 300])
        cat = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)   #([64, 300])
        # output shape => [seq_len, batch_size, hidden_size]
        # hidden shape => [num_layers, batch_size, hidden_size]
        # cell shape => [num_layers, batch_size, hidden_size]
 
        class_labels = self.W(cat)   #(batch_size, num_classes) 
        return class_labels
    
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
        
def train_classification_network(args, train_dataset, val_dataset, vocabulary, word_emd_model):
    training_start_time = time.time()

    #load model architecture 
    vocab_size = len(vocabulary)  # Size of your vocabulary
    ckpt_path = "/home/jayaram/research/NLP_course_papers_and_content/Assignments/Assign_2/logs/ELMo"
    checkpoint_path = ckpt_path + "/state_5.pt"
    # Create the model and optimizer objects
    #load model architecture 
        #load model architecture 

    # Input shape: [seq_len : 5, batch_size, embedding_dim]
    # Output shape: [seq_len, batch_size, hidden_size]
    # Hidden shape: [num_layers : 2, batch_size, hidden_size]
    bi_lstm = BiLSTM()
    print('vocab size: {}'.format(vocab_size))
    LM_model = ELMo(word_emd_model, bi_lstm, vocab_size)
    elmo_model = LM_model.to(device)

    # Load the checkpoint
    checkpoint = torch.load(checkpoint_path)
    # Access the desired variables from the checkpoint
    model_state_dict = checkpoint['model']
    # optimizer_state_dict = checkpoint['optimizer']
    # epoch = checkpoint['epoch']


    # optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
    # Load the model and optimizer states from the checkpoint
    elmo_model.load_state_dict(model_state_dict)

    embedding_dim = 600  # Dimension of word embeddings
    hidden_dim = 300  # Dimension of hidden state in LSTM
    num_layers = 1  # Number of LSTM layers
    classification_model = classification_LSTM(embedding_dim, hidden_dim, num_layers, elmo_model)
    classification_model = classification_model.to(device)


    #-----------------------------------------------------------------------------#
    #--------------------------------- main loop ---------------------------------#
    #-----------------------------------------------------------------------------#
    is_LSTM = True
    trainer = Trainer_classification_jayaram(classification_model, train_dataset, val_dataset, is_LSTM)

    for i in range(args.epochs):
        print(f'Epoch {i} / {args.epochs} | {args.save_ckpt}')
        trainer.train_classification_ELMo(device, i, n_train_steps=args.number_of_steps_per_epoch)

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
        default=200,
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

# # Custom collate function
# fixed_length =40
# def collate_fn(batch):
#     padded_batch = batch
#     # Pad sequences in the batch to have the same length
#     for i, sentence in enumerate(batch):
#         words = sentence.split(" ")
#         words = [string for string in words if string]
#         if len(words) >= fixed_length:
#             padded_batch[i] = words[:fixed_length]  # Truncate if longer
#         else:
#             num_padding = fixed_length - len(words)
#             padded_batch[i] = words + ['<pad>'] * num_padding
            
#     # padded_batch = [torch.tensor(sentence.split(" ")) for sentence in batch]
#     # padded_batch = pad_sequence(padded_batch, batch_first=True, padding_value='<pad>')
#     return padded_batch

#build vocab
csv_file = 'AG_data/train.csv'
data_train = pd.read_csv(csv_file, header=0, names=['Class Index', 'Description'])   #120k rows
# data_train['Description'] = data_train['Description'] + " ."  (every sentence already has . at the end)
texts = " ".join(data_train['Description'].tolist())   #" ".join(...): Joins the elements of the list into a single string, with each text description separated by a space.

# max_len = 0
# new_list_disc = []
# for input in data_train['Description'].tolist(): 
#     words = input.split(" ")
#     max_len = max(max_len, len(words))
#     if(len(words) < 40):     #98910
#         new_list_disc.append(input)

# Shuffle the list randomly
sentence_list = data_train['Description'].tolist()
labels_list = data_train['Class Index'].tolist()
labels_list = [x - 1 for x in labels_list]# random.shuffle(sentence_list)
# Calculate the split index (e.g., 50%)
split_index = len(sentence_list) // 6

# Split the list into two parts
val = sentence_list[:split_index]   #20k
train = sentence_list[split_index:]  #100k

# Split the original list into two parts: 'val' and 'train'
val_labels = labels_list[:split_index]   # The first 1/6th of the data (approximately 20,000 items)
train_labels = labels_list[split_index:]  #

all_data = data_train['Description'].tolist() 
max_seq_len = 10

print('total no of train sentences before filtering: {}'.format(len(train)))
print('total no of val sentences before filtering: {}'.format(len(val)))

train_max_len = 0
train_sentences = []
train_labels_filtered = []
for i, text in  enumerate(train):
    train[i] = train[i].lower()
    train[i] = train[i].strip('\n')
    train[i] = re.sub(r"(@\[A-Za-z0-9]+)|([^0-9A-Za-z \n])|(\w+:\/\/\S+)|^rt|www.+?", " ", train[i])

    words = train[i].split(" ")
    train_max_len = max(train_max_len, len(words))

    if(len(words) <= max_seq_len):
        train_sentences.append(train[i])   #89k
        train_labels_filtered.append(train_labels[i])

print('total no of train sentences after filtering: {}'.format(len(train_sentences)))

# construct vocab
corpus = " ".join(sentence_list)   #combine all sentences into single string
train_str = " ".join(train_sentences)   #combine all train sentences into single string

# process data (split train and test data into words)
corpus_words = split_into_words(corpus)  #split into words
train_data_words = split_into_words(train_str)  #split into words
print("Number of tokens in Training data = ",len(train_data_words))

# vocab = construct_vocab(train_data_words)  # give a label to each unique word 
vocab = construct_vocab(corpus_words)  # give a label to each unique word 

val_sentences = []
val_labels_filtered = []
for i, text in  enumerate(val):
    val[i] = val[i].lower()
    val[i] = val[i].strip('\n')
    val[i] = re.sub(r"(@\[A-Za-z0-9]+)|([^0-9A-Za-z \n])|(\w+:\/\/\S+)|^rt|www.+?", " ", val[i])

    words = val[i].split(" ")

    if(len(words) <= max_seq_len):
        val_data_processed = process_data(words, vocab)    # use unk token for all unseen words in test corpus
        val_sentences.append(val[i])
        val_labels_filtered.append(val_labels[i])
   
print('total no of val sentences after filtering: {}'.format(len(val_sentences)))
val_str = " ".join(val_sentences)   #combine all val sentences into single string
val_data_words = split_into_words(val_str)
print("Number of tokens in val data = ",len(val_data_words))

# if('farhang' in train_data_words):
#     print('in train data there')
# # print(vocab['farhang'])
# if('farhang' in val_data_words):
#     print('in val data')
# print(vocab['farhang'])
# vocab = Counter(train_data_words)
print("Size of Vocab",len(vocab))
 
# Create DataLoader instances
batch_size = 64
# train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
# val_loader = DataLoader(val_data, batch_size=batch_size)

# # Example usage:
# for batch in train_loader:
#     inputs = batch['text']

#     max_len = 0
#     for input in inputs: 
#         words = input.split(" ")
#         max_len = max(max_len, len(words))

#     labels = batch['label']


# Load the GloVe pretrained vectors
word_emd_model = t_vocab.GloVe(name='6B', dim=300)  # You can change 'dim' to match the dimensionality of your GloVe model
# Initialize dataset
window_size = max_seq_len
train_dataset = AGNews_Dataset(train_sentences, train_labels_filtered,  window_size, vocab, word_emd_model)
val_dataset = AGNews_Dataset(val_sentences, val_labels_filtered, window_size, vocab, word_emd_model)

# train_dataloader = torch.utils.data.DataLoader(
#         train_dataset, batch_size=256, num_workers=1, collate_fn = collate_fn, shuffle=True, pin_memory=True
#     )
# # print(next(iter(train_dataloader))) 
# for idx, train_batch in enumerate(train_dataloader):
#     print(train_batch) 

# train the ELMo model
train_classification_network(args, train_dataset, val_dataset, vocab, word_emd_model)

# max_len = 0
# for input in inputs: 
#     words = input.split(" ")
#     max_len = max(max_len, len(words))

