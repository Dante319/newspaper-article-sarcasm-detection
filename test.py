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
from keras.metrics import binary_accuracy
from keras import backend as K
from keras.engine.topology import Layer, InputSpec
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score
import sklearn
import pandas as pd
from keras.models import load_model

# load model
model = load_model('C:/Users/Dante/PycharmProjects/CIP/model.h5')
# summarize model.
model.summary()

csv_file = 'C:/Users/Dante/PycharmProjects/CIP/Datasets/processed_summ.csv'
enc = OneHotEncoder(handle_unknown='ignore')

readCSV = pd.read_csv(csv_file)
texts = readCSV['Headline']
labels = readCSV['Label']

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
print(data, labels)
# In[ ]:

indices = np.arange(data.shape[0])
np.random.shuffle(indices)
data = data[indices]
labels = labels[indices]

test, pred = [], []
predictions = model.predict(data)
for i in range(len(predictions)):
    val = [round(j) for j in predictions[i]]
    pred.append(val)
    #print(type(val))
    actual = labels[i]
    test.append(actual.tolist())
    print("Val: {}, Actual: {}".format(val, actual))
