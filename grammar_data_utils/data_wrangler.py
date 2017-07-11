import numpy
from keras.preprocessing.text import text_to_word_sequence
from nltk import FreqDist
import numpy as np
import pandas as pd
import pickle
from tensorflow.python.platform import gfile
import tensorflow as tf


import codecs


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

    except BaseException as e:
        print(e)
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



def create_vocabulary(vocabulary_path, data_path, max_vocabulary_size,
                      tokenizer=None, normalize_digits=True):
    if not gfile.Exists(vocabulary_path):
        print("Creating vocabulary %s from data %s" % (vocabulary_path, data_path))
        vocab = {}
        with gfile.GFile(data_path, mode="rb") as f:
            counter = 0
            for line in f:
                counter += 1
                if counter % 100000 == 0:
                    print("  processing line %d" % counter)

                tokens = tokenizer(line) #if tokenizer else basic_tokenizer(line)

                for word in tokens:
                    if word in vocab:
                        vocab[word] += 1
                    else:
                        vocab[word] = 1
            vocab_list = sorted(vocab, key=vocab.get, reverse=True)
            if len(vocab_list) > max_vocabulary_size:
                vocab_list = vocab_list[:max_vocabulary_size]
            with gfile.GFile(vocabulary_path, mode="wb") as vocab_file:
                for w in vocab_list:
                    w = tf.compat.as_bytes(w)
                    vocab_file.write(w + b"\n")

def initialize_vocabulary(vocabulary_path):

      if gfile.Exists(vocabulary_path):
        rev_vocab = []
        with gfile.GFile(vocabulary_path, mode="rb") as f:
            rev_vocab.extend(f.readlines())
        rev_vocab = [(line.strip()) for line in rev_vocab]
        vocab = dict([(x, y) for (y, x) in enumerate(rev_vocab)])
        return vocab, rev_vocab
      else:
        raise ValueError("Vocabulary file %s not found.", vocabulary_path)


def sentence_to_token_ids(sentence, vocabulary,
                          tokenizer=None, normalize_digits=True):

    if tokenizer:
        words = tokenizer(sentence)

    return [vocabulary.get(w) for w in words]


def data_to_token_ids(data_path, target_path, vocabulary_path,tokenizer=None, normalize_digits=True):


    if not gfile.Exists(target_path):
        print("Tokenizing data in %s" % data_path)
        vocab, _ = initialize_vocabulary(vocabulary_path)
        with gfile.GFile(data_path, mode="rb") as data_file:
            with gfile.GFile(target_path, mode="w") as tokens_file:
                counter = 0
                for line in data_file:
                    counter += 1
                    if counter % 100000 == 0:
                        print("  tokenizing line %d" % counter)
                    token_ids = sentence_to_token_ids((line), vocab,
                                            tokenizer, normalize_digits)
                    final = '{}\n'.format(token_ids)
                    tokens_file.write(final)

def new_data_to_token_ids(sentence, vocabulary_path,tokenizer=None,normalize_digits=True):
    vocab, _ = initialize_vocabulary(vocabulary_path)
    token_ids = sentence_to_token_ids((sentence), vocab,
                                            tokenizer, normalize_digits)

    return token_ids

