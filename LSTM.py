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
from sklearn.metrics import accuracy_score
import sklearn
import pandas as pd

def fun_LSTM(csv_file):
    #enc = OneHotEncoder(handle_unknown='ignore')

    readCSV = pd.read_csv(csv_file)
    texts = readCSV['headline']
    labels = readCSV['is_sarcastic']

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

    labels = to_categorical(np.asarray(labels))  # convert to one-hot encoding vectors
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

    # In[ ]:
    EMBEDDING_DIM = 100
    GLOVE_DIR = "C:/Users/Dante/PycharmProjects/CIP/Datasets/glove.twitter.27B.100d.txt"
    embeddings_index = {}
    f = open(GLOVE_DIR, encoding="UTF-8")
    print("Loading GloVe from:", GLOVE_DIR, "...", end="")
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()
    print("Done.\nProceeding with Embedding Matrix...")
    embedding_matrix = np.random.random((len(word_index) + 1, EMBEDDING_DIM))
    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector
    print("Completed!")

    embedding_layer = Embedding(len(word_index) + 1,
                                EMBEDDING_DIM,
                                weights=[embedding_matrix],
                                input_length=MAX_SEQUENCE_LENGTH,
                                trainable=False)

    sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
    embedded_sequences = embedding_layer(sequence_input)
    l_lstm1 = Bidirectional(LSTM(2,dropout=0.2,recurrent_dropout=0.2,return_sequences=True))(embedded_sequences)
    l_cov1= Conv1D(24, 9, activation='relu')(l_lstm1)
    l_pool1 = MaxPooling1D(2)(l_cov1)
    l_drop1 = Dropout(0.3)(l_pool1)
    l_flat = Flatten()(l_drop1)
    l_dense = Dense(16, activation='relu')(l_flat)
    preds = Dense(2, activation='softmax')(l_dense)

    model = Model(sequence_input, preds)
    adadelta = optimizers.Adadelta(lr=1.0, rho=0.95, epsilon=None, decay=0.0)

    # Add metric if needed
    def get_lr_metric(optimizer):
        def lr(y_true, y_pred):
            return optimizer.lr  # K.eval(optimizer.lr)

        return lr
    lr_metric = get_lr_metric(adadelta)
    model.compile(loss='categorical_crossentropy',
                  optimizer=adadelta,
                  metrics=['acc'])

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

    tensorboard = callbacks.TensorBoard(log_dir='./logs', histogram_freq=0, batch_size=16, write_grads=True , write_graph=True)
    model_checkpoints = callbacks.ModelCheckpoint("checkpoint.h5", monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=False, mode='auto', period=1)
    lr_schedule = callbacks.LearningRateScheduler(step_cyclic)

    model.summary()
    print("Training Progress:")
    history = model.fit(x_train, y_train, validation_data=(x_val, y_val),
                        epochs=30, batch_size=64,
                        callbacks=[lr_schedule])

    # evaluate the model
    scores = model.evaluate(x_test, y_test, verbose=0)
    print("%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))

    model.save("Dam/model.h5")
    print("Saved model to disk")

    test, pred = [], []
    predictions = model.predict(x_test)
    for i in range(len(predictions)):
        val = [round(j) for j in predictions[i]]
        pred.append(val)
        actual = y_test[i]
        test.append(actual)
        print("Val: {}, Actual: {}".format(val, actual))


filename = 'C:/Users/Dante/PycharmProjects/CIP/Datasets/convertcsv.csv'.format("headlines")
fun_LSTM(filename)