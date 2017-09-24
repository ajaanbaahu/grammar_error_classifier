#!/Users/rmohan/flaskapp/lib/python3.4

import sys
sys.path.insert(0, '/home/paperspace/workspace/develop-dir/grammar-error-check')

from grammar_models.model_seq import read_data, simple_LSTM
import fileinput
import numpy as np
import tensorflow as tf
from keras.preprocessing import sequence
import pandas as pd
from grammar_data_utils.data_wrangler import generate_arrays_from_file,convert_to_df
from keras.callbacks import ModelCheckpoint
import ast

tf.app.flags.DEFINE_string("correct_data",None,"path to correct training data")
tf.app.flags.DEFINE_string("incorrect_data",None,"path to incorrect training data")
tf.app.flags.DEFINE_string("correct_dev_data",None,"path to correct dev data")
tf.app.flags.DEFINE_string("incorrect_dev_data",None,"path to incorrect dev data")
tf.app.flags.DEFINE_integer("train_samples",1000,'samples to train')
tf.app.flags.DEFINE_string("test_samples",500,'sampels to test')
tf.app.flags.DEFINE_integer("max_seq_len",200,'size of data point')
tf.app.flags.DEFINE_integer("size", 100, "Size of each model layer.")
tf.app.flags.DEFINE_integer("vocab_size",300000,"size of vocabulary")
tf.app.flags.DEFINE_integer("batch_size",1024,"size of invidiual batches")
tf.app.flags.DEFINE_integer("embedding_size",64,"embed vector len")
tf.app.flags.DEFINE_integer("epochs",20,"number of iterations")
tf.app.flags.DEFINE_integer("yield_size",1000,"size of samples yo yield")



FLAGS = tf.app.flags.FLAGS
sess = tf.Session()
from keras import backend as K
K.set_session(sess)

def train():


    correct_source_file   = FLAGS.correct_data
    incorrect_source_file = FLAGS.incorrect_data
    correct_dev_data = FLAGS.correct_dev_data
    incorrect_dev_data = FLAGS.incorrect_dev_data
    train_samples = FLAGS.train_samples
    test_samples = FLAGS.test_samples
    max_seq_length = FLAGS.max_seq_len
    embedding_vector_length = FLAGS.embedding_size
    # max_review_length = FLAGS.max_seq_len
    num_hidden_units =  FLAGS.size
    batch_size = FLAGS.batch_size
    iterations = FLAGS.epochs
    yield_size = FLAGS.yield_size
    counter = 0 

    with sess.as_default():
        print("Creating model with %d unites" %num_hidden_units)
        model = simple_LSTM(embedding_vector_length,max_seq_length,num_hidden_units)
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

        training_data = generate_arrays_from_file(correct_source_file,incorrect_source_file,max_seq_length,yield_size)
        
        with open(correct_dev_data,'r') as fh1, open(incorrect_dev_data,'r') as fh2:
        

            data1 = [ast.literal_eval(j) for j in fh1 if 'None' not in j]
            label_corr = np.ones(len(data1))
            data2 = [ast.literal_eval(j) for j in fh2 if 'None' not in j]
            label_mod = np.zeros(len(data2))
            temp = convert_to_df(data1,label_corr,100000)
            temp2 = convert_to_df(data2,label_mod,100000)

        df_full = pd.concat([temp,temp2],axis=0)
        df_full = df_full.dropna(axis=0)
        X_val = df_full['sent_ids'].as_matrix()
        y_val = df_full['label'].as_matrix()
        X_val = sequence.pad_sequences(X_val,max_seq_length)

        print("training model")
        filepath="weights.best.hdf5"
        checkpoint = ModelCheckpoint(filepath, save_best_only=True, mode='max', verbose=1)
        callbacks_list = [checkpoint]
        
        for x,y in training_data:
            counter += yield_size
            
            model.fit(x,y, epochs = iterations, batch_size=batch_size,callbacks=callbacks_list)
            model.evaluate(X_val,y_val, batch_size=64, verbose=6)

            if counter > train_samples/2:
                print("exhauted everything", 10*'*',counter)
                break

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