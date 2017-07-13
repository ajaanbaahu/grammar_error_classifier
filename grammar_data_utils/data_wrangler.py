import numpy
from keras.preprocessing.text import text_to_word_sequence
from keras.preprocessing import sequence
from nltk import FreqDist
import numpy as np
import pandas as pd
import pickle
from tensorflow.python.platform import gfile
import tensorflow as tf
import ast

import codecs

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



    '''for very large files'''

def generate_arrays_from_file(correct_grammar_file, incorrect_grammar_file , max_lines, max_pad_length):
    counter = 0
    print(correct_grammar_file,incorrect_grammar_file,max_lines,max_pad_length)

    with open(correct_grammar_file,'r') as fh1, open(incorrect_grammar_file,'r') as fh2:
        while 1:
            for x,y in zip(fh1, fh2):

                if 'None' in x:
                    continue
                if 'None' in y:
                    continue
                counter +=1

                # if  counter == max_lines:
                #     break
                try:

                    x = sequence.pad_sequences([ast.literal_eval(x)],max_pad_length),np.array([[1]])
                    y = sequence.pad_sequences([ast.literal_eval(y)],max_pad_length),np.array([[0]])


                    yield x

                except TypeError as e:
                    print(e)