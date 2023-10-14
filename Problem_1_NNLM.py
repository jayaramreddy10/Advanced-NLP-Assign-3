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

# Set CUDA_LAUNCH_BLOCKING environment variable
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

random.seed(32)

from Trainer import Trainer_jayaram

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# wandb setup
number = 1   #experiment number
NAME = "model" + str(number)
ID = 'NNLM_training_' + str(number)
run = wandb.init(project='NNLM_training', name = NAME, id = ID)

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
        pred_word = self.data[idx + self.window_size]
        label = self.vocab[pred_word]   #vocab is dict (maps words to labels)
 
        #get prefix
        prefix_window = self.data[idx: idx + self.window_size]
        prefix_word_embeddings = []

        # # Define the <UNK> token to represent out-of-vocabulary words
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
        prefix_emb = torch.from_numpy(prefix_emb)

        return prefix_emb, torch.tensor(label)
    
class NNLM(nn.Module):
    def __init__(self, input_dim, hidden_dim_1, hidden_dim_2, len_vocab, window_size, dropout_prob):
        super(NNLM, self).__init__() # init the base Module Class
        self.input_dim = input_dim
        self.hidden_dim_1 = hidden_dim_1
        self.hidden_dim_2 = hidden_dim_2
        self.len_vocab = len_vocab
        self.dropout_prob = dropout_prob

        self.tanh = torch.nn.Tanh()

        self.Relu = torch.nn.ReLU()
        self.dropout = nn.Dropout(p=self.dropout_prob)  # Dropout layer with the specified probability
        self.W1 = nn.Linear(input_dim*window_size, hidden_dim_1)
        self.W2 = nn.Linear(hidden_dim_1, hidden_dim_2)
        self.W3 = nn.Linear(hidden_dim_2, len_vocab)

    def forward(self, input): 
        # hidden1 = self.tanh(self.W1(input))
        # hidden2 = self.tanh(self.W2(hidden1))
        hidden1 = self.tanh(self.W1(input)) 
        hidden1 = self.dropout(hidden1) 
        hidden2 = self.tanh(self.W2(hidden1))
        hidden2 = self.dropout(hidden2)

        out = self.W3(hidden2)
        # return out 
        # probs = torch.nn.functional.softmax(out, dim=1)
        return out # probs
    
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
        
def train_network(args, train_dataset, val_dataset):
    training_start_time = time.time()

    #load model architecture 
    LM_model = NNLM(input_dim = 300, hidden_dim_1 = 200, hidden_dim_2 = 25, len_vocab=len(vocabulary), window_size = 5)
    LM_model = LM_model.to(device)

    #-----------------------------------------------------------------------------#
    #--------------------------------- main loop ---------------------------------#
    #-----------------------------------------------------------------------------#

    trainer = Trainer_jayaram(LM_model, train_dataset, val_dataset)
    for i in range(args.epochs):
        print(f'Epoch {i} / {args.epochs} | {args.save_ckpt}')
        trainer.train_NNLM(device, i, n_train_steps=args.number_of_steps_per_epoch)

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
        default=200,  #5000
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
    nlp_sentencizer.max_length = 6400000
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
    train_network(args, train_dataset, val_dataset)

    # report perplexity scores
    # compute_perplexity(test_dataset)
