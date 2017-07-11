import sys
sys.path.insert(0, '/Users/rmohan/grammar-error-check')
from grammar_data_utils.data_wrangler import process_input,convert_to_ids, convert_seq2ids, convert_to_df, dump_to_pickle, read_from_pickle, create_vocabulary, data_to_token_ids, new_data_to_token_ids
import pickle
import pandas as pd
import numpy as np
from keras.preprocessing.text import text_to_word_sequence
from tensorflow.python.platform import gfile
import tensorflow as tf


#input files for correct and incorrect grammar data
file2 = '/Users/rmohan/sequenceModels/data/modulated_dev_en.txt'
file1 = '/Users/rmohan/sequenceModels/data/newstest2009.en'

max_vocabulary_size = 100000
#destination files for processed data
create_vocabulary('new_vocab',file2, max_vocabulary_size, tokenizer=text_to_word_sequence, normalize_digits=True)
data_to_token_ids(file1,'id_file %s'%'corr','new_vocab',tokenizer=text_to_word_sequence)
data_to_token_ids(file2,'id_file %s'%'mod','new_vocab',tokenizer=text_to_word_sequence)


