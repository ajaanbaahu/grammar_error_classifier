import sys
sys.path.insert(0, '/home/paperspace/workspace/develop-dir/grammar-error-check')
from grammar_data_utils.data_wrangler import convert_to_df, dump_to_pickle, read_from_pickle, create_vocabulary, data_to_token_ids, new_data_to_token_ids
import pickle
import pandas as pd
import numpy as np
from keras.preprocessing.text import text_to_word_sequence
from tensorflow.python.platform import gfile
from keras.preprocessing import sequence
import ast
#import tensorflow as tf



#input files for correct and incorrect grammar data
file2 = '/Users/rmohan/sequenceModels/data/modulated_dev_en.txt'
file1 = '/Users/rmohan/sequenceModels/data/newstest2009.en'

# max_vocabulary_size = 100000
# #destination files for processed data
# create_vocabulary('new_vocab',file2, max_vocabulary_size, tokenizer=text_to_word_sequence, normalize_digits=True)
# data_to_token_ids(file1,'id_file %s'%'corr','new_vocab',tokenizer=text_to_word_sequence)
# data_to_token_ids(file2,'id_file %s'%'mod','new_vocab',tokenizer=text_to_word_sequence)

file_cor = '/home/paperspace/workspace/grammar_classification_data/corr_dev_grammar.ids'
file_mod = '/home/paperspace/workspace/grammar_classification_data/mod_dev_grammar.ids'

with open(file_cor,'r') as fh1, open(file_mod,'r') as fh2:
	coll = []

	# for j in fh1:
	# 	if j and j is not 'None':
	# 		coll.append(ast.literal_eval(j))
	#coll = sequence.pad_sequences(coll,200)

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
X_val = sequence.pad_sequences(X_val,200)

print((X_val[:5]))

