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
    
class AugusteDataset(Dataset):
    def __init__(self, data, window_size, vocab, word_emb_model, is_train = True, transform = None):
        self.is_train = is_train
        self.transform = transform
        self.window_size = window_size
        self.data = data    # list fo words
        self.n_words = len(data)
        self.vocab = vocab
        self.word_emb_model = word_emb_model

    def __len__(self):
        return self.n_words - self.window_size

    def __getitem__(self, idx):
        #get label
        label_window = self.data[idx + 1: idx + self.window_size + 1]
        # pred_word = self.data[idx + self.window_size]
        # print('label window: {}'.format(label_window))
        labels = [self.vocab[word]  for word in label_window]  #vocab is dict (maps words to labels)
        # print(labels)
 
        #get prefix
        prefix_window = self.data[idx: idx + self.window_size]
        prefix_word_embeddings = []

        # Define the <UNK> token to represent out-of-vocabulary words
        unk_token = 'unk'
        # Iterate through the words in the window
        for word in prefix_window:
            if word in self.word_emb_model.stoi:
                # If the word is in the GloVe vocabulary, get its embedding
                word_embedding = self.word_emb_model.vectors[self.word_emb_model.stoi[word]]
            else:
                # If the word is not in the GloVe vocabulary, use the embedding of <UNK>
                word_embedding = self.word_emb_model.vectors[self.word_emb_model.stoi[unk_token]]
            prefix_word_embeddings.append(word_embedding)

        # prefix_word_embeddings = [self.word_emb_model[word] for word in prefix_window]
        prefix_emb = np.concatenate(prefix_word_embeddings)
        prefix_emb = np.reshape(prefix_emb, (self.window_size, -1))   #(seq_len, embed_dim for LSTM)

        labels = np.array(labels)
        return prefix_emb, labels
    
class LSTM_LM(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, n_layers, vocab_size, bidirectional = False):
        super().__init__()
        # self.embedding = nn.Embedding(input_dim, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=n_layers, bidirectional=bidirectional)
        self.W = nn.Linear(hidden_dim, vocab_size)

    def forward(self, input):
        # input shape => [seq_len, batch_size, embedding_dim]
        output, (hidden, cell) = self.lstm(input)
        # output shape => [seq_len, batch_size, hidden_size]
        # hidden shape => [num_layers, batch_size, hidden_size]
        # cell shape => [num_layers, batch_size, hidden_size]
 
        output = self.W(output)   #(seq_len, batch_size, vocab_size) , will broadcasting work here?
        return output, (hidden, cell)
        
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

    # Input shape: [seq_len : 5, batch_size, embedding_dim]
    # Output shape: [seq_len, batch_size, hidden_size]
    # Hidden shape: [num_layers : 2, batch_size, hidden_size]
    LM_model = LSTM_LM(embedding_dim, hidden_dim, num_layers, vocab_size)
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

    # split text into sentences
    nlp_sentencizer = spacy.blank("en")
    nlp_sentencizer.add_pipe("sentencizer")
    nlp_sentencizer.max_length = 6351665
    corpus = open('Auguste_Maquet.txt', 'r')
    ds = corpus.read()
    ds.replace("\n", "")
    tokens = nlp_sentencizer(ds)

    sentences = []
    for sentence in tokens.sents:
        sentences.append(str(sentence).replace('\n', " "))
    print(len(sentences))

    # split sentences to train and text
    # random.shuffle(sentences)
    train = sentences[:30000]
    val = sentences[30000:40000]
    test = sentences[40000:59710]
    train = ''.join(str(train))   #combine all train sentences into single string
    val = ''.join(str(val))   #combine all train sentences into single string
    test = ''.join(str(test))     #combine all test sentences  into single string

    # process data (split train and test data into words)
    train_data = split_into_words(train)  #split into words
    val_data = split_into_words(val)
    test_data = split_into_words(test)
    vocabulary = construct_vocab(train_data)  # give a label to each unique word 
    train_vocab_len = len(vocabulary)
    vocabulary['unk'] = train_vocab_len # 1 for safety
    val_data_processed = process_data(val_data, vocabulary)  # use <UNK> token for all unseen words in test corpus
    test_data_processed = process_data(test_data, vocabulary)  # use <UNK> token for all unseen words in test corpus
    # print(len(test_data_processed))

    # get word embeddings from pretrained glove.6B file
    # glove_file_path = "/home/jayaram/research/NLP_course_papers_and_content/Assignments/Assign_1/pretrained_word_embeddings/glove.6B.300d.txt"
    # word2vec_output_file = 'glove.6B.300d.txt.word2vec'# Create an instance of the GloveWordEmbeddings class
    # glove_embeddings = GloveWordEmbeddings(glove_file_path, word2vec_output_file)
    # # Convert GloVe to Word2Vec format
    # glove_embeddings.convert_to_word2vec_format()
    # # Load the Word2Vec model
    # word_emd_model = glove_embeddings.load_word2vec_model()   # we can get embeddings from this model

    # Load the GloVe pretrained vectors
    word_emd_model = t_vocab.GloVe(name='6B', dim=300)  # You can change 'dim' to match the dimensionality of your GloVe model

    # Initialize dataset
    window_size = 5
    train_dataset = AugusteDataset(train_data, window_size, vocabulary, word_emd_model)
    val_dataset = AugusteDataset(val_data_processed, window_size, vocabulary, word_emd_model)
    test_dataset = AugusteDataset(test_data_processed, window_size, vocabulary, word_emd_model)

    # train the NNLM model
    train_network(args, train_dataset, val_dataset, vocabulary)

    # report perplexity scores
    # compute_perplexity(test_dataset)
