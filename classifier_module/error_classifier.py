import numpy
#from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.preprocessing.text import text_to_word_sequence
from nltk import FreqDist
import numpy as np
import pandas as pd
import pickle
# fix random seed for reproducibility
numpy.random.seed(7)

from keras.models import model_from_json

from grammar_models.model_seq import prediction_LSTM
from grammar_data_utils.convert2ids import newSeq2id

embedding_vec_len = 128
max_review_length = 200

model = prediction_LSTM(embedding_vec_len,max_review_length)

model.load_weights('/Users/rmohan/sequenceModels/error-detection/model.h5')

print("Loaded model from disk")

# evaluate loaded model on test data
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

data = newSeq2id("Students and teachers are taking the date lightly.")

new_s = sequence.pad_sequences(data, maxlen=max_review_length)

a = np.reshape(new_s,(1,max_review_length))

print(model.predict(a))

import gc;gc.collect()