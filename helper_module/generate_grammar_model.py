#!/Users/rmohan/flaskapp/lib/python3.4

import sys
sys.path.insert(0, '/Users/rmohan/grammar-error-check')

from grammar_models.model_seq import read_data, simple_LSTM
import fileinput
import numpy as np
import tensorflow as tf
from keras.preprocessing import sequence
import pandas as pd
from grammar_data_utils.data_wrangler import generate_arrays_from_file
from keras.callbacks import ModelCheckpoint

tf.app.flags.DEFINE_string("correct_data",None,"path to correct training data")
tf.app.flags.DEFINE_string("incorrect_data",None,"path to incorrect training data")
tf.app.flags.DEFINE_string("correct_dev_data",None,"path to correct dev data")
tf.app.flags.DEFINE_string("incorrect_dev_data",None,"path to incorrect dev data")
tf.app.flags.DEFINE_integer("train_samples",1000,'samples to train')
tf.app.flags.DEFINE_string("test_samples",500,'sampels to test')
tf.app.flags.DEFINE_integer("max_seq_len",20,'size of data point')
tf.app.flags.DEFINE_integer("size", 100, "Size of each model layer.")
tf.app.flags.DEFINE_integer("vocab_size",300000,"size of vocabulary")
tf.app.flags.DEFINE_integer("batch_size",1024,"size of invidiual batches")
tf.app.flags.DEFINE_integer("embedding_size",64,"embed vector len")
tf.app.flags.DEFINE_integer("epochs",20,"number of iterations")


FLAGS = tf.app.flags.FLAGS


def train():


    correct_source_file   = FLAGS.correct_data
    incorrect_source_file = FLAGS.incorrect_data
    correct_dev_data = FLAGS.correct_dev_data
    incorrect_dev_data = FLAGS.incorrect_dev_data
    train_samples = FLAGS.train_samples
    test_samples = FLAGS.test_samples
    max_seq_length = FLAGS.max_seq_len
    embedding_vector_length = FLAGS.embedding_size
    max_review_length = FLAGS.max_seq_len
    num_hidden_units =  FLAGS.size
    batch_size = FLAGS.batch_size
    iterations = FLAGS.epochs


    #with tf.Session() as sess:
    print("Creating model with %d unites" %num_hidden_units)
    model = simple_LSTM(embedding_vector_length,max_seq_length,num_hidden_units)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    training_data = generate_arrays_from_file(correct_source_file,incorrect_source_file,train_samples,max_review_length)

    validation_samples = generate_arrays_from_file(correct_dev_data,incorrect_dev_data,test_samples,max_review_length)
    print("training model")
    filepath="weights.best.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    callbacks_list = [checkpoint]
    model.fit_generator(training_data,validation_data=validation_samples,steps_per_epoch = batch_size, epochs = iterations, verbose=4,validation_steps=batch_size)
    #
    '''serialize model to JSON'''
    model_json = model.to_json()
    with open("model.json", "w") as json_file:
        json_file.write(model_json)
        '''serialize weights to HDF5'''
        model.save_weights("model.h5")
        print("Saved model to disk")
def main():

    train()

if __name__ == "__main__":
    main()