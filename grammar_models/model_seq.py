import numpy
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
# fix random seed for reproducibility
numpy.random.seed(7)
import pickle
import pandas as pd
from sklearn.utils import shuffle
import optparse


top_words = 300000
def read_data(source_file, train_samples, test_sample,max_review_length):
    print(source_file,train_samples,test_sample,max_review_length)

    try:
        with open(source_file,'rb') as fh:
            data = pickle.load(fh)
            data = shuffle(data)
            X_train, y_train, X_test, y_test = data['sent_ids'][:train_samples], data['label'][:train_samples],data['sent_ids'][-test_sample:], data['label'][-test_sample:]

            X_train = X_train.as_matrix()
            y_train = y_train.as_matrix()
            X_test  = X_test.as_matrix()
            y_test = y_test.as_matrix()

            X_train = sequence.pad_sequences(X_train, maxlen=max_review_length)
            X_test = sequence.pad_sequences(X_test, maxlen=max_review_length)
        return X_train,y_train,X_test,y_test
    except:
        return 0,0,0,0
        #X_train, y_train, X_test, y_test = None,None,None,None



def simple_LSTM(embedding_vector_length,max_review_length,num_hidden_units):
	model = Sequential()
	model.add(Embedding(top_words, embedding_vector_length, input_length=max_review_length))
	model.add(LSTM(num_hidden_units))
	model.add(Dense(1, activation='sigmoid'))
	#model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
	print('Type : Simple LSTM')
	print(model.summary())
	return model


def LSTM_dropput(embedding_vector_length,max_review_length,num_hidden_units):
	model = Sequential(embedding_vector_length,max_review_length)
	model.add(Embedding(top_words, embedding_vector_length, input_length=max_review_length))
	model.add(LSTM(num_hidden_units,dropout=0.2,recurrent_dropout=0.2))
	model.add(Dense(1, activation='sigmoid'))
	#model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
	print('Type : LSTM_Dropout')
	#print(model.summary())
	return model


def prediction_LSTM(embedding_vector_length,max_review_length,num_hidden_units):
	model = Sequential()
	model.add(Embedding(top_words, embedding_vector_length, input_length=max_review_length))
	model.add(LSTM(num_hidden_units))
	model.add(Dense(1, activation='sigmoid'))
	#model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
	print('Type : Prediction LSTM')
	print(model.summary())
	return model

