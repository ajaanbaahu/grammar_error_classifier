import numpy
from keras.preprocessing.text import text_to_word_sequence
from nltk import FreqDist
import numpy as np
import pandas as pd
import pickle


import codecs

#input files for correct and incorrect grammar data
# file2 = '/Users/rmohan/sequenceModels/data/modulated_en.txt'
# file1 = '/Users/rmohan/sequenceModels/data/correct_en.txt'
#
# #destination files for processed data
# corr_sentid_file = '../data/grammar_classification_data/correct_ids.pkl'
# mod_sentid_file = '../data/grammar_classification_data/mod_ids.pkl'


def tokenize_sequence(sent_seq):
    return [text_to_word_sequence(sents.replace('\u2028',' ')) for sents in sent_seq.split('\n')]

def process_input(file, vocab_size):
    f = codecs.open(file, encoding='utf-8')


    raw_data = f.read()
    print(len(raw_data))
    tokenised = [text_to_word_sequence(sents.replace('\u2028',' ')) for sents in raw_data.split('\n')]
    try:
        dist = FreqDist(np.hstack(tokenised))
        vocab = dist.most_common(vocab_size -1)
        # data_len = len(tokenised)
        # if flag == 1:
        #     y = np.ones(data_len)
        # elif flag == 0:
        #     y = np.zeros(data_len)
    except BaseException as e:
        print(e)
    #print(len(tokenised), len(y), len(vocab))
    return tokenised, vocab

def convert_to_ids(word_sequence, vocab):

    print("this is stage 2", len(word_sequence))
    ix_to_word = [word[0] for word in vocab]
    word_to_ix = {word:ix for ix, word in enumerate(ix_to_word)}
    print(len(word_to_ix))

    return ix_to_word,word_to_ix

def convert_seq2ids(word_seq, ix_to_w, w_to_ix):


    for i, sentence in enumerate(word_seq):

        for j, word in enumerate(sentence):
            if word in w_to_ix:
                
                word_seq[i][j] = w_to_ix[word]

    return word_seq


def convert_to_df(sent_ids, sent_label,n):
    temp = pd.DataFrame(list(zip(sent_ids,sent_label))[:n],columns=['sent_ids','label'])
    return temp

def dump_to_pickle(file_path,data):

    with open(file_path,'wb') as fh:
        pickle.dump(data,fh)
    print("done writing files")

def read_from_pickle(file_path):
    with open(file_path,'rb') as fh:
        data = pickle.load(fh)
        return data


#tokenize sentences and generate vocab data for correct grammar
# corr_data, corr_vocab_data = process_input(file1, vocab_size=1000000)
#
# #generate correct word2ids, id2word for correct grammar
# id2w_C,w2id_C = convert_to_ids(corr_data, corr_vocab_data)
#
# #generate word2ids sequence for correct grammar
# final_correct_seq_ids = convert_seq2ids(corr_data,id2w_C,w2id_C)
# #give lables to data
# corr_label = np.ones(len(final_correct_seq_ids))
#
# #tokenize sentences and generate vocab data for incorrect grammar
# mod_data, mod_vocab_data = process_input(file2, vocab_size=1000000)
#
# #generate correct word2ids, id2word for incorrect grammar
# id2w_M,w2id_M = convert_to_ids(mod_data, mod_vocab_data)
#
# #generate word2ids sequence for incorrect grammar
# final_mod_seq_ids = convert_seq2ids(mod_data,id2w_M,w2id_M)
# #give lables to data
# mod_label = np.zeros(len(final_mod_seq_ids))
#
# ''' dump and read data if needed '''
# #dump_to_pickle(corr_sentid_file, final_correct_seq_ids)
# #dump_to_pickle(mod_sentid_file, final_mod_seq_ids)
#
#
# #final_correct_seq_ids = read_from_pickle(corr_sentid_file)
# #final_mod_seq_ids = read_from_pickle(mod_sentid_file)
#
#
# #size of dataframe
# n = 20000000
#
# #dataframe for correct grammar
# df_correct = convert_to_df(final_correct_seq_ids, corr_label,n)
#
# #dataframe for incorrect grammar
# df_modulated =convert_to_df(final_mod_seq_ids, mod_label,n)
#
# final_mod_seq_ids,final_correct_seq_ids,mod_label, corr_label = '','','',''
#
# #combing the two dataframe
# df_full = pd.concat([df_correct,df_modulated],axis=0)
#
# full_df_file = '../data/grammar_classification_data/full_df.pkl'
#
# # dump_to_pickle(full_df_file,df_full)
# dump_to_pickle(full_df_file,df_full)
#
