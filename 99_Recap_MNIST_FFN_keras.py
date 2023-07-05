'''
Data Engineering
'''

'''
D01. Import Libraries for Data Engineering
'''
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split

'''
D02. Load MNIST data / Only for Toy Project
'''
mnist = tf.keras.datasets.mnist
(X_train, Y_train), (X_test, Y_test) = mnist.load_data()

'''
D03. Split data
'''
X_train, X_valid, Y_train, Y_valid = \
    train_test_split(X_train, Y_train, test_size=0.2)

'''
D04. EDA(? / Exploratory data analysis)
'''

# plot first few images
for i in range(9):
    # define subplot
    plt.subplot(330 + 1 + i)
    # plot raw pixel data
    plt.imshow(X_train[i], cmap=plt.get_cmap('gray'))
    # if you want to invert color, you can use 'gray_r'. this can be used only for MNIST, Fashion MNIST not cifar10
    # pyplot.imshow(trainX[i], cmap=pyplot.get_cmap('gray_r'))
    
# show the figure
plt.show()


'''
D05. Data Preprocessing
'''
# Change data type as float. If it is int type, it might cause error 
# Flattening & Normalizing
X_train = (X_train.reshape(-1, 784) / 255).astype(np.float32)
X_test  = (X_test.reshape(-1, 784) / 255).astype(np.float32)
X_valid = (X_valid.reshape(-1, 784) / 255).astype(np.float32)

# One-Hot Encoding
Y_train = np.eye(10)[Y_train].astype(np.float32)
Y_test  = np.eye(10)[Y_test].astype(np.float32)
Y_valid = np.eye(10)[Y_valid].astype(np.float32)

'''
Model Engineering
'''

'''
M01. Import Libraries for Model Engineering
'''

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from tensorflow.keras.models import load_model

'''
M02. [PASS] TPU Initialization
'''

'''
M03. Define Hyperparameters for Model Engineering
'''

hidden_size = 256
output_dim = 10      # output layer dimensionality = num_classes
EPOCHS = 100
batch_size = 100
learning_rate = 5e-4

'''
M04. [PASS] Open "strategy.scope(  )"
'''

'''
M05. Build NN model
'''
model = tf.keras.models.Sequential()
model.add(Dense(hidden_size, activation='sigmoid'))
model.add(Dense(hidden_size, activation='sigmoid'))
model.add(Dense(hidden_size, activation='sigmoid'))
model.add(Dense(output_dim, activation='softmax'))

'''
class Feed_Forward_Net(Model):
    
    # Multilayer perceptron
    
    def __init__(self, hidden_size, output_dim):
        super().__init__()
        self.l1 = Dense(hidden_size, activation='sigmoid')
        self.l2 = Dense(hidden_size, activation='sigmoid')
        self.l3 = Dense(hidden_size, activation='sigmoid')
        self.l4 = Dense(output_dim, activation='softmax')

    def call(self, x):
        h1 = self.l1(x)
        h2 = self.l2(h1)
        h3 = self.l3(h2)
        y = self.l4(h3)

        return y

model = Feed_Forward_Net(hidden_size, output_dim)
'''

'''
M06. Optimizer
'''

# optimizer = optimizers.SGD(learning_rate=learning_rate)
optimizer = optimizers.Adam(learning_rate=learning_rate)

'''
M07. Model Compilation - model.compile
'''
model.compile(optimizer=optimizer, loss='categorical_crossentropy',
              metrics = ['accuracy'])

'''
M08. EarlyStopping
'''
es = EarlyStopping(monitor = 'val_loss', mode = 'min', verbose = 1, patience = 8)

'''
M09. ModelCheckpoint
'''
mc = ModelCheckpoint('best_model.h5', monitor = 'val_accuracy', mode = 'max', verbose = 1, save_best_only = True)

'''
M10. Train and Validation - `model.fit`
'''
history = model.fit(X_train, Y_train, epochs = EPOCHS,
                    batch_size=batch_size,
                    validation_data = (X_valid, Y_valid),
                    verbose=1,
                    callbacks=[es, mc])

'''
M11. Assess model performance
'''
loaded_model = load_model('best_model.h5')
print("\n Test Accuracy: %.4f" % (loaded_model.evaluate(X_test, Y_test)[1]))

'''
M12. [Opt] Plot Loss and Accuracy
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
M13. [PASS] Training result test for Code Engineering
'''

