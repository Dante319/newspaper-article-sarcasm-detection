
# coding: utf-8

# ## CNN Text Classification with Keras

# In[1]:
import numpy as np
import csv
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from keras import regularizers, initializers, optimizers, callbacks
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical
from keras.layers import Embedding
from keras.layers import Dense, Input, Flatten
from keras.layers import Conv1D, MaxPooling1D, Embedding, Dropout, LSTM, GRU, Bidirectional
from keras.models import Model
from keras import backend as K
from keras.engine.topology import Layer, InputSpec
from sklearn.preprocessing import OneHotEncoder
import sklearn
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import pandas as pd
from pathlib import Path

def KerasCNN(csv_file):
    # Read from dataset (`data.csv` file)

    # In[ ]:

    enc = OneHotEncoder(handle_unknown='ignore')

    readCSV = pd.read_csv(csv_file)
    texts = readCSV['headline']
    labels = readCSV['is_sarcastic']


    # Model Parameters

    # In[ ]:
    MAX_NB_WORDS = 40000
    MAX_SEQUENCE_LENGTH = 30
    VALIDATION_SPLIT = 0.2

    # In[ ]:


    tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)
    word_index = tokenizer.word_index
    print('Found %s unique tokens.' % len(word_index))
    data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)


    # In[ ]:


    labels = to_categorical(np.asarray(labels)) # convert to one-hot encoding vectors
    print('Shape of data tensor:', data.shape)
    print('Shape of label tensor:', labels.shape)


    # In[ ]:


    indices = np.arange(data.shape[0])
    np.random.shuffle(indices)
    data = data[indices]
    labels = labels[indices]
    nb_validation_samples = int(VALIDATION_SPLIT * data.shape[0])


    # In[ ]:
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(data, labels, test_size=0.2, random_state=1) #split into train and test
    x_train, x_val, y_train, y_val = sklearn.model_selection.train_test_split(x_train, y_train, test_size=0.2, random_state=1) #split train into train and validation

    #x_train = data[:-nb_validation_samples]
    #y_train = labels[:-nb_validation_samples]
    #x_val = data[-nb_validation_samples:]
    #y_val = labels[-nb_validation_samples:]

    print('Number of entries in each category:')
    print("Training:\n", y_train.sum(axis=0))
    print("Validation:\n", y_val.sum(axis=0))


    # ### Preparing the Embedding layer
    # Compute an index mapping words to known embeddings, by parsing the data dump of pre-trained embeddings: [GloVe](https://nlp.stanford.edu/projects/glove/) vectors from Stanford NLP. For new words, a "randomised vector" will be created.

    # In[ ]:

    EMBEDDING_DIM = 100
    GLOVE_DIR = "C:/Users/Dante/PycharmProjects/CIP/Datasets/glove.twitter.27B.100d.txt"
    glove = open(GLOVE_DIR, encoding="UTF-8")
    print("Loading GloVe from:", GLOVE_DIR, "...", end="")
    embeddings_index = {}
    for line in glove:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    glove.close()
    print("Done.\nProceeding with Embedding Matrix...")
    embedding_matrix = np.random.random((len(word_index) + 1, EMBEDDING_DIM))
    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector
    print("Completed!")


    # After computing our embedding matrix, load this embedding matrix into an `Embedding` layer. Toggle `trainable=False` to prevent the weights from being updated during training.

    # In[ ]:


    embedding_layer = Embedding(len(word_index) + 1,
                                EMBEDDING_DIM,
                                weights=[embedding_matrix],
                                input_length=MAX_SEQUENCE_LENGTH,
                                trainable=False)


    # ### Simplified 1D CNN
    # [Reference](https://github.com/richliao/textClassifier), [Source](https://blog.keras.io/using-pre-trained-word-embeddings-in-a-keras-model.html) and [Notes](http://www.wildml.com/2015/11/understanding-convolutional-neural-networks-for-nlp/)

    # In[ ]:


    sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32') # 1000
    embedded_sequences = embedding_layer(sequence_input)
    l_cov1= Conv1D(128, 3, activation='relu',kernel_regularizer=regularizers.l2(0.01))(embedded_sequences)
    l_cov2= Conv1D(128, 3, activation='relu',kernel_regularizer=regularizers.l2(0.01))(l_cov1)
    l_pool1 = MaxPooling1D(3)(l_cov2)
    #l_drop1 = Dropout(0.2)(l_pool1)
    l_cov3 = Conv1D(128, 3, activation='relu',kernel_regularizer=regularizers.l2(0.01))(l_pool1)
    l_pool4 = MaxPooling1D(6)(l_cov3) # global max pooling
    l_flat = Flatten()(l_pool4)
    l_dense = Dense(128, activation='relu')(l_flat)
    preds = Dense(2, activation='softmax')(l_dense) # Modify categories based on dataset


    # In[ ]:


    model = Model(sequence_input, preds)
    adadelta = optimizers.Adadelta(lr=1.0, rho=0.95, epsilon=None, decay=0)
    model.compile(loss='categorical_crossentropy',
                  optimizer=adadelta,
                  metrics=['acc'])
    model.summary()

    # In[ ]:


    def step_cyclic(epoch):
        try:
            if epoch%11==0:
                return float(2)
            elif epoch%3==0:
                return float(1.1)
            elif epoch%7==0:
                return float(0.6)
            else:
                return float(1.0)
        except:
            return float(1.0)

    tensorboard = callbacks.TensorBoard(log_dir='./logs', histogram_freq=0, batch_size=256, write_grads=True , write_graph=True)
    model_checkpoints = callbacks.ModelCheckpoint("checkpoints", monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=False, mode='auto', period=1)
    lr_schedule = callbacks.LearningRateScheduler(step_cyclic)


    # In[ ]:

    print("Training Progress:")
    history = model.fit(x_train, y_train, validation_data=(x_val, y_val),
              epochs=10, batch_size=128,
              callbacks=[tensorboard, model_checkpoints, lr_schedule])

    # evaluate the model
    scores = model.evaluate(x_test, y_test, verbose=0)
    print("%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))

    test, pred = [], []
    predictions = model.predict(x_test)
    for i in range(len(predictions)):
        val = [round(j) for j in predictions[i]]
        pred.append(val)
        actual = y_test[i]
        test.append(actual)
        print("Val: {}, Actual: {}".format(val, actual))


filename = 'C:/Users/Dante/PycharmProjects/CIP/Datasets/convertcsv.csv'.format("headlines")
KerasCNN(filename)