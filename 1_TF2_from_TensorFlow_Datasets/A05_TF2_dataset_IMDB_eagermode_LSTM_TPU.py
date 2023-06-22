'''
A. Data Engineering
'''

'''
1. Import Libraries for Data Engineering
'''
import re
import numpy as np
import matplotlib.pyplot as plt

'''
2. Import IMDB Dataset from library
'''
from tensorflow.keras.datasets import imdb

'''
T. TPU Initialization
'''
import tensorflow as tf

try:
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
    print('Running on TPU {}'.format(tpu.cluster_spec().as_dict()['worker']))
except ValueError:
    tpu = None

if tpu:
    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)
    strategy = tf.distribute.experimental.TPUStrategy(tpu)
else:
    strategy = tf.distribute.get_strategy()

print("REPLICAS: {}".format(strategy.num_replicas_in_sync))

print('Loading data...')
vocab_size = 20000
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=vocab_size)
# (X_train, y_train), (X_test, y_test) = imdb.load_data()

'''
3. Explore the several features of datasets.
'''
print('Number of reviews for training : {}'.format(len(X_train)))
print('Number of reviews for tesing   : {}'.format(len(X_test)))
num_classes = len(set(y_train))
print('Number of Classes : {}'.format(num_classes))

unique_elements, counts_elements = np.unique(y_train, return_counts=True)
print("Frequency for each label:")
print(np.asarray((unique_elements, counts_elements)))

'''
4. Tokenizer and Vocab define
'''
word_to_index = imdb.get_word_index()
index_to_word={}
for key, value in word_to_index.items():
    index_to_word[value+3] = key

print('Top #1 word in frequency : {}'.format(index_to_word[4]))

print('Top 3938 most frequent words : {}'.format(index_to_word[3941]))

for index, token in enumerate(("<pad>", "<sos>", "<unk>")):
    index_to_word[index]=token

print(' '.join([index_to_word[index] for index in X_train[0]]))
print(' '.join([index_to_word[index] for index in X_train[0][:50]]))

'''
5. Tokenizer test
'''
lines = [
  "It is winter and the weather is very cold.",
  "Will this Christmas be a white Christmas?",
  "Be careful not to catch a cold in winter and have a happy new year."
]
for line in lines:
    # txt_2_ids = ' '.join([index_to_word[index] for index in line])
    new_sentence = re.sub('[^0-9a-zA-Z ]', '', line).lower()

    # 정수 인코딩
    encoded = []
    for word in new_sentence.split():
        try :
            encoded.append(word_to_index[word]+3)

        # 단어 집합에 없는 단어는 <unk> 토큰으로 변환.
        except KeyError:
            encoded.append(2)
    ids_2_txt = ' '.join([index_to_word[index] for index in encoded])
    print("Input     :", line)
    print("txt_2_ids :", encoded)
    print("ids_2_txt :", ids_2_txt,"\n")

'''
6. Explore the tokenized datasets.
'''
len_result = [len(s) for s in X_train]

print('Maximum length of review : {}'.format(np.max(len_result)))
print('Average length of review : {}'.format(np.mean(len_result)))

plt.subplot(1,2,1)
plt.boxplot(len_result)
plt.subplot(1,2,2)
plt.hist(len_result, bins=50)
plt.show()

print(X_train[0])
print(y_train[0])

batch_size = 32

'''
7. Split Data
'''
# print('Loading data...')
# (X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=vocab_size)

X_valid = X_test[:20000]
y_valid = y_test[:20000]

X_test = X_test[20000:]
X_pred = X_test
y_test = y_test[20000:]

print(len(X_train), 'train sequences')
print(len(X_valid), 'valid sequences')
print(len(X_test) , 'test sequences')

'''
8. Pad sequences
'''
from tensorflow.keras.preprocessing.sequence import pad_sequences

max_len = 300  # cut texts after this number of words (among top vocab_size most common words)
X_train = pad_sequences(X_train, maxlen=max_len)
X_valid = pad_sequences(X_valid, maxlen=max_len)
X_test  = pad_sequences(X_test, maxlen=max_len)

# 10. Data type define
# For IMDB dataset set, it is not required.

# 11. Build dataset
# For eager mode, it is done at the "model.fit"

'''
B. Model Engineering
'''

'''
9. Import Libraries for Model Engineering
'''
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Embedding
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from tensorflow.keras.models import load_model

'''
10. Set Hyperparameters
'''
embedding_dim = 256
hidden_units = 128
EPOCHS = 20
learning_rate = 5e-4

# initialize and compile model within strategy scope
with strategy.scope():
    '''
    11. Build NN model
    '''
    model = Sequential()
    model.add(Embedding(vocab_size, embedding_dim))
    model.add(LSTM(hidden_units, dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(1, activation='sigmoid'))

    '''
    12. Optimizer
    '''
    optimizer = optimizers.Adam(learning_rate=learning_rate)

    '''
    13. Model Compilation - model.compile
    '''
    # model.compile(optimizer='adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    model.compile(optimizer=optimizer, loss = 'binary_crossentropy', metrics = ['accuracy'])

model.summary()

'''
14. EarlyStopping
'''
es = EarlyStopping(monitor = 'val_loss', mode = 'min', verbose = 1, patience = 8)

'''
15. ModelCheckpoint
'''
mc = ModelCheckpoint('best_model.h5', monitor = 'val_accuracy', mode = 'max', verbose = 1, save_best_only = True)

'''
16. Train and Validation - `model.fit`
'''
history = model.fit(X_train, y_train, epochs = EPOCHS, validation_data = (X_valid, y_valid), callbacks=[es, mc])

'''
17. Assess model performance
'''
loaded_model = load_model('best_model.h5')
print("\n 테스트 정확도: %.4f" % (loaded_model.evaluate(X_test, y_test)[1]))

'''
18. [Opt] Plot Loss and Accuracy
'''
history_dict = history.history
history_dict.keys()

acc      = history_dict['accuracy']
val_acc  = history_dict['val_accuracy']
loss     = history_dict['loss']
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

'''
19. [Opt] Training result test for Code Engineering
'''
def sentiment_predict(new_sentence):
    # 알파벳과 숫자를 제외하고 모두 제거 및 알파벳 소문자화
    new_sentence = re.sub('[^0-9a-zA-Z ]', '', new_sentence).lower()

    # 정수 인코딩
    encoded = []
    for word in new_sentence.split():
        try :
            # 단어 집합의 크기를 10,000으로 제한.
            if word_to_index[word] <= vocab_size:
                encoded.append(word_to_index[word]+3)
            else:
            # 10,000 이상의 숫자는 <unk> 토큰으로 변환.
                encoded.append(2)
        # 단어 집합에 없는 단어는 <unk> 토큰으로 변환.
        except KeyError:
            encoded.append(2)

    pad_sequence = pad_sequences([encoded], maxlen=max_len) # 패딩
    score = float(loaded_model.predict(pad_sequence)) # 예측

    if(score > 0.5):
        print("A positive review with a {:.2f}% chance. ".format(score * 100))
    else:
        print("A negative review with a {:.2f}% chance. ".format((1 - score) * 100))

for idx in range(10):
    print('----'*30)
    test_input = ' '.join([index_to_word[index] for index in X_pred[idx]])
    print("Test sentence from datasets:\n", test_input)
    sentiment_predict(test_input)
    if(y_test[idx] > 0.5):
        print("Ground truth is positive!")
    else:
        print("Ground truth is negative!")
    
    
