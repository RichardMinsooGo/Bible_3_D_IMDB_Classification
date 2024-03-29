'''
A. Data Engineering
'''

'''
D1. Import IMDB Raw Dataset from Auther's gdrive
'''

!pip install --upgrade --no-cache-dir gdown

from IPython.display import clear_output 
clear_output()

# Mini-Imagenet dataset download from Auther's Github repository
import gdown

google_path = 'https://drive.google.com/uc?id='
file_id = '1tqZpPvvyluyu7VVvk99tKpJd4cIkS4yi'
output_name = 'IMDB_Dataset.csv'
gdown.download(google_path+file_id,output_name,quiet=False)
#https://drive.google.com/file/d/1tqZpPvvyluyu7VVvk99tKpJd4cIkS4yi/view?usp=sharing

'''
D2. Install torchtext Libraries
'''
!pip install -U torchtext==0.10.0

'''
D3. Import Libraries for Data Engineering
'''

import torch
from torchtext.legacy import data
from torchtext.legacy import datasets
# import torch.nn.functional as F

import re
import numpy as np
from sklearn.model_selection import train_test_split    

import random
SEED = 1234
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

'''
D4. Tokenizer Install & import
''' 
# Spacy Tokenizer is default. So, You are no need to install it.

'''
D5. Load and modifiy to pandas dataframe
'''
import pandas as pd

pd.set_option('display.max_colwidth', 100)
# pd.set_option('display.max_colwidth', None)

dataset_df = pd.read_csv('/content/IMDB_Dataset.csv')

print(len(dataset_df))

dataset_df.head()

dataset_df.rename(columns = {'review':'SRC', 'sentiment':'TRG'}, inplace = True)
dataset_df.head()

dataset_df['TRG'] = dataset_df['TRG'].str.replace('positive','1')
dataset_df['TRG'] = dataset_df['TRG'].str.replace('negative','0')
dataset_df.head()

# dataset_df['TRG'] = dataset_df['TRG'].astype(int)
# TRG_df = dataset_df["TRG"].to_numpy()
# print(TRG_df[:10])
# len(TRG_df)

'''
D6. Preprocess and build list
'''
def preprocess_func(sentence):
    sentence = sentence.lower().strip()
    # creating a space between a word and the punctuation following it
    # eg: "he is a boy." => "he is a boy ."
    sentence = re.sub(r"([?.!,])", r" \1 ", sentence)
    sentence = re.sub(r'[" "]+', " ", sentence)
    # removing contractions
    sentence = re.sub(r"i'm", "i am", sentence)
    sentence = re.sub(r"he's", "he is", sentence)
    sentence = re.sub(r"she's", "she is", sentence)
    sentence = re.sub(r"it's", "it is", sentence)
    sentence = re.sub(r"that's", "that is", sentence)
    sentence = re.sub(r"what's", "that is", sentence)
    sentence = re.sub(r"where's", "where is", sentence)
    sentence = re.sub(r"how's", "how is", sentence)
    sentence = re.sub(r"\'ll", " will", sentence)
    sentence = re.sub(r"\'ve", " have", sentence)
    sentence = re.sub(r"\'re", " are", sentence)
    sentence = re.sub(r"\'d", " would", sentence)
    sentence = re.sub(r"\'re", " are", sentence)
    sentence = re.sub(r"won't", "will not", sentence)
    sentence = re.sub(r"can't", "cannot", sentence)
    sentence = re.sub(r"n't", " not", sentence)
    sentence = re.sub(r"n'", "ng", sentence)
    sentence = re.sub(r"'bout", "about", sentence)
    # replacing everything with space except (a-z, A-Z, ".", "?", "!", ",")
    sentence = re.sub(r"[^a-zA-Z?.!,]+", " ", sentence)
    sentence = sentence.strip()
    return sentence

dataset_df['SRC'] = dataset_df['SRC'].apply(preprocess_func)

'''
D7. Split Data
'''
train_data, test_data = train_test_split(dataset_df, test_size=0.2, random_state=32)
valid_data, test_data = train_test_split(test_data, test_size=0.2, random_state=32)

print(len(train_data))
print(len(valid_data))
print(len(test_data))
print(train_data.shape)
print(valid_data.shape)
print(test_data.shape)

'''
D8. Tokenizer define
'''

TEXT = data.Field(tokenize = 'spacy',
                  tokenizer_language = 'en',
                  batch_first = True, fix_length= 300)

LABEL = data.LabelField(dtype = torch.float)

'''
D9. Tokenizer test
# PASS
'''

'''
D10. Pad sequence
# PASS
'''

'''
D11. Convert dataset
'''
def convert_dataset(input_data, text, label):
    list_of_example = [data.Example.fromlist(row.tolist(), fields=[('text', text), ('label', label)])  for _, row in input_data.iterrows()]
    dataset = data.Dataset(examples=list_of_example, fields=[('text', text), ('label', label)])
    return dataset

train_data = convert_dataset(train_data, TEXT, LABEL)
valid_data = convert_dataset(valid_data, TEXT, LABEL)
test_data  = convert_dataset(test_data, TEXT, LABEL)

print(f'Number of training examples   : {len(train_data)}')
print(f'Number of validation examples : {len(valid_data)}')
print(f'Number of testing examples    : {len(test_data)}')


'''
D12. Build vocaburary
'''
MAX_VOCAB_SIZE = 20000

TEXT.build_vocab(train_data, max_size = MAX_VOCAB_SIZE)

LABEL.build_vocab(train_data)

print(f"Unique tokens in TEXT vocabulary : {len(TEXT.vocab)}")
print(f"Unique tokens in LABEL vocabulary: {len(LABEL.vocab)}")

print(TEXT.vocab.freqs.most_common(20))

print(TEXT.vocab.itos[:10])

print(LABEL.vocab.stoi)

'''
D13. Dataload with Iterator
# PASS
'''

'''
D14. Data type define
# PASS
'''

'''
D15. Dataload with Iterator
'''
BATCH_SIZE = 64

train_iterator, valid_iterator, test_iterator = data.Iterator.splits(
    (train_data, valid_data, test_data),
    batch_size = BATCH_SIZE,
    sort = False,
    device = device)

print('Number of minibatch for training dataset   : {}'.format(len(train_iterator)))
print('Number of minibatch for validation dataset : {}'.format(len(valid_iterator)))
print('Number of minibatch for testing dataset    : {}'.format(len(test_iterator)))

'''
B. Model Engineering
'''

'''
M1. Import Libraries for Model Engineering
'''
import torch.nn as nn


'''
M2. Set Hyperparameters
'''
embedding_dim = 256
hidden_units = 128
EPOCHS = 50
learning_rate = 5e-4

'''
M4. Build NN model
'''
class LSTM(nn.Module):
    def __init__(self, vocab_size, hidden_dim, output_dim, embedding_dim, dropout):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.rnn = nn.LSTM(embedding_dim, hidden_dim, batch_first=True, dropout=dropout)
        self.linear = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, text):
        embedded = self.dropout(self.embedding(text))
        output, _ = self.rnn(embedded)
        output = self.linear(output[:, -1, :])
        return output
  
    def _init_state(self, batch_size=1):
        weight = next(self.parameters()).data
        return weight.new(self.n_layers, batch_size, self.hidden_dim).zero_()

model = LSTM(len(TEXT.vocab), 128, len(LABEL.vocab)-1, 300, 0.2)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f'The model has {count_parameters(model):,} trainable parameters')

'''
M5. Optimizer
'''
import torch.optim as optim
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

'''
M6. Define Loss Function
'''
criterion = nn.BCEWithLogitsLoss()

model = model.to(device)
criterion = criterion.to(device)

'''
M7. Define Accuracy Function
'''
def binary_accuracy(preds, target):
    '''
    from https://github.com/bentrevett/pytorch-sentiment-analysis/blob/master/1%20-%20Simple%20Sentiment%20Analysis.ipynb
    '''
    # round predictions to the closest integer (0 or 1)
    rounded_preds = torch.round(torch.sigmoid(preds))

    #convert into float for division
    correct = (rounded_preds == target).float()

    # rounded_preds = [ 1   0   0   1   1   1   0   1   1   1]
    # targets       = [ 1   0   1   1   1   1   0   1   1   0]
    # correct       = [1.0 1.0 0.0 1.0 1.0 1.0 1.0 1.0 1.0 0.0]
    acc = correct.sum() / len(correct)
    return acc

'''
M8. Define Training Function
'''
def train(model, iterator, optimizer, criterion):
    
    epoch_loss = 0
    epoch_acc = 0
    
    model.train()
    
    for batch in iterator:
        
        # We initialize the gradient to 0 for every batch.
        optimizer.zero_grad()

        # batch of sentences인 batch.text를 model에 입력
        predictions = model(batch.text).squeeze(1)
        
        # Calculate the loss value by comparing the prediction result with batch.label 
        loss = criterion(predictions, batch.label)

        # Accuracy calculation
        acc = binary_accuracy(predictions, batch.label)
        
        # Backpropagation using backward()
        loss.backward()

        # Update the parameters using the optimization algorithm
        optimizer.step()
        
        epoch_loss += loss.item()
        epoch_acc += acc.item()
        
    return epoch_loss / len(iterator), epoch_acc / len(iterator)

'''
M9. Define Validation / Test Function
'''
def evaluate(model, iterator, criterion):
    
    epoch_loss = 0
    epoch_acc = 0
    
    # "evaluation mode" : turn off "dropout" or "batch nomalizaation"
    model.eval()

    # Use less memory and speed up computation by preventing gradients from being computed in pytorch
    with torch.no_grad():
    
        for batch in iterator:
            
            predictions = model(batch.text).squeeze(1)
            
            loss = criterion(predictions, batch.label)
            
            acc = binary_accuracy(predictions, batch.label)

            epoch_loss += loss.item()
            epoch_acc += acc.item()
        
    return epoch_loss / len(iterator), epoch_acc / len(iterator)

import time

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

best_valid_loss = float('inf')

'''
M10. Episode / each step Process
'''
for epoch in range(EPOCHS):

    start_time = time.time()
    
    train_loss, train_acc = train(model, train_iterator, optimizer, criterion)
    valid_loss, valid_acc = evaluate(model, valid_iterator, criterion)
    
    end_time = time.time()

    epoch_mins, epoch_secs = epoch_time(start_time, end_time)
    
    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), 'tut4-model.pt')
    
    print(f'Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
    print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%')
    print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc*100:.2f}%')

'''
M11. Assess model performance (Test step)
'''
model.load_state_dict(torch.load('tut4-model.pt'))

test_loss, test_acc = evaluate(model, test_iterator, criterion)

print(f'Test Loss: {test_loss:.3f} | Test Acc: {test_acc*100:.2f}%')

import torch
model.load_state_dict(torch.load('tut4-model.pt'))

'''
M12. [Opt] Training result test for Code Engineering
'''
import spacy
nlp = spacy.load('en_core_web_sm')

def predict_sentiment(model, sentence, min_len = 5):
    model.eval()
    tokenized = [tok.text for tok in nlp.tokenizer(sentence)]
    if len(tokenized) < min_len:
        tokenized += ['<pad>'] * (min_len - len(tokenized))
    indexed = [TEXT.vocab.stoi[t] for t in tokenized]
    tensor = torch.LongTensor(indexed).to(device)
    tensor = tensor.unsqueeze(0)
    prediction = torch.sigmoid(model(tensor))
    return prediction.item()

examples = [
  "This film is terrible",
  "This film is great",
  "This movie is fantastic"
]

for idx in range(len(examples)) :

    sentence = examples[idx]
    pred = predict_sentiment(model,sentence)
    print("\n",sentence)
    if pred >= 0.5 :
        print(f">>>This is a positive review. ({pred : .2f})")
    else:
        print(f">>>This is a negative review.({pred : .2f})")

