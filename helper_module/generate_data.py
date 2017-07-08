from grammar_data_utils.data_wrangler import process_input,convert_to_ids, convert_seq2ids, convert_to_df, dump_to_pickle, read_from_pickle
import pickle
import pandas as pd
import numpy as np


#input files for correct and incorrect grammar data
file2 = '/Users/rmohan/sequenceModels/data/newsdev2014.en'
file1 = '/Users/rmohan/sequenceModels/data/correct_dev_en.txt'

#destination files for processed data
corr_sentid_file = '/Users/rmohan/sequenceModels/data/grammar_classification_data/correct_ids.pkl'
mod_sentid_file = '/Users/rmohan/sequenceModels/data/grammar_classification_data/mod_ids.pkl'

vocab_file = '/Users/rmohan/sequenceModels/data/grammar_classification_data/vocab.pkl'


#tokenize sentences and generate vocab data for correct grammar
corr_data, corr_data_vocab = process_input(file1, vocab_size=1000000)

#generate correct word2ids, id2word for correct grammar
id2w_C,w2id_C = convert_to_ids(corr_data, corr_data_vocab)

#generate word2ids sequence for correct grammar
final_correct_seq_ids = convert_seq2ids(corr_data,id2w_C,w2id_C)
#give lables to data
corr_label = np.ones(len(final_correct_seq_ids))

#tokenize sentences and generate vocab data for incorrect grammar
mod_data, mod_data_vocab = process_input(file2, vocab_size=1000000)

#generate correct word2ids, id2word for incorrect grammar
id2w_M,w2id_M = convert_to_ids(mod_data, mod_data_vocab)

#generate word2ids sequence for incorrect grammar
final_mod_seq_ids = convert_seq2ids(mod_data,id2w_M,w2id_M)
#give lables to data
mod_label = np.zeros(len(final_mod_seq_ids))

''' dump and read data if needed '''


dump_to_pickle(vocab_file, corr_data_vocab)
dump_to_pickle(corr_sentid_file, final_correct_seq_ids)
dump_to_pickle(mod_sentid_file, final_mod_seq_ids)


# final_correct_seq_ids = read_from_pickle(corr_sentid_file)
# final_mod_seq_ids = read_from_pickle(mod_sentid_file)


#size of dataframe
n = 20000000

#dataframe for correct grammar
df_correct = convert_to_df(final_correct_seq_ids, corr_label,n)

#dataframe for incorrect grammar
df_modulated =convert_to_df(final_mod_seq_ids, mod_label,n)

final_mod_seq_ids,final_correct_seq_ids,mod_label, corr_label = '','','',''

#combing the two dataframe
df_full = pd.concat([df_correct,df_modulated],axis=0)

full_df_file = '/Users/rmohan/sequenceModels/data/grammar_classification_data/full_df.pkl'

# dump_to_pickle(full_df_file,df_full)
dump_to_pickle(full_df_file,df_full)
#