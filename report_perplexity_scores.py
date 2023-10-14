import numpy as np
import math
import spacy
import torch
import re
import torch.nn as nn
import torchtext.vocab as t_vocab

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
    
class LSTM_LM(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, n_layers, vocab_size, bidirectional = False):
        super().__init__()
        # self.embedding = nn.Embedding(input_dim, embedding_dim)   --not required (we get it from Glove)
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
        text = re.sub(r"(@\[A-Za-z0-9]+)|([^0-9A-Za-z \n.,])|(\w+:\/\/\S+)|^rt|www.+?", " ", text)

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
        

def get_word_embeddings(sentence_words, word_emb_model):
    #get prefix
    prefix_word_embeddings = []

    # # Define the <UNK> token to represent out-of-vocabulary words
    unk_token = 'unk'
    # Iterate through the words in the window
    for word in sentence_words:
        if word in word_emb_model.stoi:
            # If the word is in the GloVe vocabulary, get its embedding
            word_embedding = word_emb_model.vectors[word_emb_model.stoi[word]]
        else:
            # If the word is not in the GloVe vocabulary, use the embedding of <UNK>
            word_embedding = word_emb_model.vectors[word_emb_model.stoi[unk_token]]
        prefix_word_embeddings.append(word_embedding)

    # prefix_word_embeddings = [self.word_emb_model[word] for word in prefix_window]
    prefix_emb = np.concatenate(prefix_word_embeddings)
    prefix_emb = torch.from_numpy(prefix_emb)

    return prefix_emb

def compute_perplexity_scores(model, dataset, word_emb_mode, vocabulary, is_NNLM = True):
    vocab_size = len(vocabulary)

    perplexity_scores_sentences = []
    for sent_no, sentence in enumerate(dataset):
        # Initialize the entropy
        entropy = 0.0
        # Tokenize and convert words to indices (replace with your preprocessing)
        words = split_into_words(sentence)  #split sentence into words
        words_processed = process_data(words, vocabulary) 
        if(len(words_processed) <= 5):
            continue
        # = [word_emb_model[word] for word in words]  # Convert words to indices
        # Iterate through the words in the segment
        for i in range(len(words_processed) - 5):  # Assuming your model uses 5 words to predict the next
            context = words_processed[i:i+5]  # Get the previous 4 words
            next_word_gt = words_processed[i+5]  # Get the next word
            if 'unk' in context:
                continue
            if(next_word_gt == 'unk') : 
                continue
            next_word_gt_label = vocabulary[next_word_gt]

            # Convert to PyTorch tensor
            inputs = torch.Tensor(get_word_embeddings(context, word_emb_mode)).unsqueeze(0)  # Add batch dimension
            inputs = inputs.to(device)#.type(dtype)
            
            # Forward pass
            if(is_NNLM):
                outputs = model(inputs)
            else:
                outputs, _ = model(inputs)   #should map from (B=1, 1500) to (B, len(vocab)) which are dist of probs

            # # Use your language model to get the probability of the next word
            probs = torch.nn.functional.softmax(outputs[0])
            probability = probs[next_word_gt_label]        
            # Define target (first word of next sentence)
            # target_word = dataset[sent_no + 1][0]
            # label = vocabulary[target_word]   #vocab is dict (maps words to labels)

            # # Calculate the entropy
            entropy += -math.log2(probability)
            # Calculate loss
            # loss = nn.CrossEntropyLoss(outputs.view(-1, vocab_size), label.view(-1))
        
        # Calculate perplexity
        perplexity = 2**(entropy / (len(words_processed) - 5))
        # perplexity = torch.exp(loss)
        # Print or store perplexity for the current sentence
        # print(f"Perplexity for sentence '{sentence}': {perplexity}")
        perplexity_scores_sentences.append(perplexity)

    return perplexity_scores_sentences

# Define your sentences and corresponding perplexity scores
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
vocabulary['unk'] = len(vocabulary)
word_emd_model = t_vocab.GloVe(name='6B', dim=300)  # You can change 'dim' to match the dimensionality of your GloVe model

folder_path = "/home/jayaram/research/NLP_course_papers_and_content/Assignments/Assign_1/logs/LSTM/hidden_300" 
is_NNLM = False
if(is_NNLM):
    # load model
    checkpoint_path = folder_path + "/state_10.pt"
    # Create the model and optimizer objects
    #load model architecture 
    model = NNLM(input_dim = 300, hidden_dim_1 = 300, hidden_dim_2 = 300, len_vocab=len(vocabulary), window_size = 5, dropout_prob=0.3)
    model = model.to(device)
else: 
    checkpoint_path = folder_path + "/state_10.pt"
    # Create the model and optimizer objects
    #load model architecture 
        #load model architecture 
    vocab_size = len(vocabulary)  # Size of your vocabulary
    embedding_dim = 300  # Dimension of word embeddings
    hidden_dim = 300  # Dimension of hidden state in LSTM
    num_layers = 1  # Number of LSTM layers

    # Input shape: [seq_len : 5, batch_size, embedding_dim]
    # Output shape: [seq_len, batch_size, hidden_size]
    # Hidden shape: [num_layers : 2, batch_size, hidden_size]
    LM_model = LSTM_LM(embedding_dim, hidden_dim, num_layers, vocab_size)
    model = LM_model.to(device)

# Load the checkpoint
checkpoint = torch.load(checkpoint_path)
# Access the desired variables from the checkpoint
model_state_dict = checkpoint['model']
# optimizer_state_dict = checkpoint['optimizer']
# epoch = checkpoint['epoch']


# optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
# Load the model and optimizer states from the checkpoint
model.load_state_dict(model_state_dict)

train = sentences[:30000]
val = sentences[30000:40000]
test = sentences[40000:59710]
# compute PPL Scores
# train
train_PPL_scores = compute_perplexity_scores(model, train, word_emd_model, vocabulary, is_NNLM)
# Calculate the average perplexity score
average_score = sum(train_PPL_scores) / len(train_PPL_scores)
# Define the file path where you want to save the text file
file_path = folder_path + "/perplexity_scores_train_NNLM.txt"
# # Open the file in write mode and write the information
with open(file_path, "w") as file:
    for sentence, score in zip(train, train_PPL_scores):
         
        if(score > 700): 
            file.write(f'{sentence}\t{score:.2f}\t\t{">700"}\n')
        else: 
            file.write(f'{sentence}\t{score:.2f}\n')

    file.write('-----------------------------------------------------------------------------\n')
    file.write(f'Average train PPL Score: {average_score:.2f}')
# Print the average score
print(f'Average train PPL Score: {average_score:.2f}')

# val
val_PPL_scores = compute_perplexity_scores(model, val, word_emd_model, vocabulary, is_NNLM)
# Calculate the average perplexity score
average_score = sum(val_PPL_scores) / len(val_PPL_scores)
# Define the file path where you want to save the text file
file_path = folder_path + "/perplexity_scores_val_NNLM.txt"
# Open the file in write mode and write the information
with open(file_path, "w") as file:
    for sentence, score in zip(val, val_PPL_scores):
        if(score > 700): 
            file.write(f'{sentence}\t{score:.2f}\t\t{">700"}\n')
        else: 
            file.write(f'{sentence}\t{score:.2f}\n')

    file.write('-----------------------------------------------------------------------------\n')
    file.write(f'Average val PPL Score: {average_score:.2f}')
# Print the average score
print(f'Average val PPL Score: {average_score:.2f}')

# test
test_PPL_scores = compute_perplexity_scores(model, test, word_emd_model, vocabulary, is_NNLM)
# Calculate the average perplexity score
average_score = sum(test_PPL_scores) / len(test_PPL_scores)
# Define the file path where you want to save the text file
file_path = folder_path + "/perplexity_scores_test_NNLM.txt"
# Open the file in write mode and write the information
with open(file_path, "w") as file:
    for sentence, score in zip(test, test_PPL_scores):
        if(score > 700): 
            file.write(f'{sentence}\t{score:.2f}\t\t{">700"}\n')
        else: 
            file.write(f'{sentence}\t{score:.2f}\n')

    file.write('-----------------------------------------------------------------------------\n')
    file.write(f'Average test PPL Score: {average_score:.2f}')
# Print the average score
print(f'Average test PPL Score: {average_score:.2f}')