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

import re
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import unicodedata

print("Tensorflow version {}".format(tf.__version__))
tf.random.set_seed(1234)
AUTO = tf.data.experimental.AUTOTUNE

# 1. Tokenizer Install & import
# Keras Tokenizer는 tensorflow 2.X 에서 기본으로 제공하는 tokenizer이며, word level tokenizer이다. 이는 별도의 설치가 필요 없다.

# 2. Copy or load raw data to Colab
max_len = 300
BATCH_SIZE  = 128
BUFFER_SIZE = 20000

import urllib3
import zipfile
import shutil
import pandas as pd

pd.set_option('display.max_colwidth', 100)
# pd.set_option('display.max_colwidth', None)

dataset_df = pd.read_csv('/content/IMDB_Dataset.csv')

print(len(dataset_df))

# dataset_df = dataset_df.loc[:, 'SRC':'TRG']
    
dataset_df.head()

dataset_df.rename(columns = {'review':'SRC', 'sentiment':'TRG'}, inplace = True)
dataset_df.head()

dataset_df['TRG'] = dataset_df['TRG'].str.replace('positive','1')
dataset_df['TRG'] = dataset_df['TRG'].str.replace('negative','0')
dataset_df.head()

dataset_df['TRG'] = dataset_df['TRG'].astype(int)

TRG_df = dataset_df["TRG"].to_numpy()

print(TRG_df[:10])

len(TRG_df)

# 5. Preprocess and build list
raw_src = []
for sentence in dataset_df['SRC']:
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
    raw_src.append(sentence)

print(raw_src[:10])

# 6. Tokenizer and Vocab define
SRC_df = pd.DataFrame(raw_src)

SRC_df.rename(columns={0: "SRC"}, errors="raise", inplace=True)

raw_src_df  = SRC_df['SRC']

print(raw_src_df[:10])

src_sentence  = raw_src_df.apply(lambda x: "<SOS> " + str(x))

filters = '!"#$%&()*+,-./:;=?@[\\]^_`{|}~\t\n'
oov_token = '<unk>'

# Define tokenizer
SRC_tokenizer = tf.keras.preprocessing.text.Tokenizer(filters = filters, oov_token=oov_token)

SRC_tokenizer.fit_on_texts(src_sentence)

vocab_size = len(SRC_tokenizer.word_index) + 1

print('Encoder 단어 집합의 크기 :',vocab_size)

# 7. Tokenizer test
lines = [
  "It is winter and the weather is very cold.",
  "Will this Christmas be a white Christmas?",
  "Be careful not to catch a cold in winter and have a happy new year."
]
for line in lines:
    txt_2_ids = SRC_tokenizer.texts_to_sequences([line])
    ids_2_txt = SRC_tokenizer.sequences_to_texts(txt_2_ids)
    print("Input     :", line)
    print("txt_2_ids :", txt_2_ids)
    print("ids_2_txt :", ids_2_txt[0],"\n")

# 8. Tokenize    
# 토큰화 / 정수 인코딩 / 시작 토큰과 종료 토큰 추가 / 패딩
tokenized_inputs  = SRC_tokenizer.texts_to_sequences(src_sentence)

len_result = [len(s) for s in tokenized_inputs]

print('Maximum length of review : {}'.format(np.max(len_result)))
print('Average length of review : {}'.format(np.mean(len_result)))

plt.subplot(1,2,1)
plt.boxplot(len_result)
plt.subplot(1,2,2)
plt.hist(len_result, bins=50)
plt.show()

# 9. Pad sequences
# 패딩
tkn_sources = tf.keras.preprocessing.sequence.pad_sequences(tokenized_inputs,  maxlen=max_len, padding='post', truncating='post')


# 10. Data type define
tkn_sources = tf.cast(tkn_sources, dtype=tf.int64)

# 11. Check tokenized data
print('질문 데이터의 크기(shape) :', tkn_sources.shape)

# 0번째 샘플을 임의로 출력
print(tkn_sources[0])

X_train = tkn_sources[:25000]
y_train = TRG_df[:25000]
X_valid = tkn_sources[25000:45000]
y_valid = TRG_df[25000:45000]
X_test  = tkn_sources[45000:]
y_test  = TRG_df[45000:]

from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, Dropout, Conv1D, GlobalMaxPooling1D 
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from tensorflow.keras.models import load_model

embedding_dim = 256
hidden_units = 128

model = Sequential()
model.add(Embedding(vocab_size, embedding_dim))
model.add(Dropout(0.3))
model.add(Conv1D(256, 3, padding='valid', activation='relu'))
model.add(GlobalMaxPooling1D())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

model.summary()
es = EarlyStopping(monitor = 'val_loss', mode = 'min', verbose = 1, patience = 8)
mc = ModelCheckpoint('best_model.h5', monitor = 'val_accuracy', mode = 'max', verbose = 1, save_best_only = True)
history = model.fit(X_train, y_train, epochs = 20, validation_data = (X_valid, y_valid), callbacks=[es, mc])

loaded_model = load_model('best_model.h5')
print("\n 테스트 정확도: %.4f" % (loaded_model.evaluate(X_test, y_test)[1]))

history_dict = history.history
history_dict.keys()

acc = history_dict['accuracy']
val_acc = history_dict['val_accuracy']
loss = history_dict['loss']
val_loss = history_dict['val_loss']

epochs = range(1, len(acc) + 1)

# "bo" is for "blue dot"
plt.plot(epochs, loss, 'o', color='g', label='Training loss')   # 'bo'
# b is for "solid blue line"
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()


plt.plot(epochs, acc, 'o', color='g', label='Training acc')   # 'bo'
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')

plt.show()

def sentiment_predict(new_sentence):
    # 알파벳과 숫자를 제외하고 모두 제거 및 알파벳 소문자화
    new_sentence = re.sub('[^0-9a-zA-Z ]', '', new_sentence).lower()

    txt_2_ids = SRC_tokenizer.texts_to_sequences([new_sentence])

    pad_sequence = pad_sequences(txt_2_ids, maxlen=max_len) # 패딩
    score = float(loaded_model.predict(pad_sequence)) # 예측

    if(score > 0.5):
        print("A positive review with a {:.2f}% chance. ".format(score * 100))
    else:
        print("A negative review with a {:.2f}% chance. ".format((1 - score) * 100))

for idx in range(10):
    print('----'*30)
    test_input = src_sentence[45000+idx]
    print("Test sentence from datasets:\n", test_input)
    sentiment_predict(test_input)
    if(y_test[idx] > 0.5):
        print("Ground truth is positive!")
    else:
        print("Ground truth is negative!")
    
    