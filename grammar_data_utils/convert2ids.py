from grammar_data_utils.data_wrangler import process_input, convert_seq2ids,convert_to_ids, dump_to_pickle, read_from_pickle, tokenize_sequence

#input files for correct and incorrect grammar data
file2 = '/Users/rmohan/sequenceModels/data/modulated_en.txt'
file1 = '/Users/rmohan/sequenceModels/data/correct_en.txt'

new_seq = tokenize_sequence('This is not a good sentence.')

id2w = read_from_pickle('/Users/rmohan/sequenceModels/grammar-error-check/grammar_data_utils/id2w.pkl')
w2id = read_from_pickle('/Users/rmohan/sequenceModels/grammar-error-check/grammar_data_utils/w2id')

#
vocab_size = 300000
#
# text_seq, vocab = process_input(file1,vocab_size)
#
# dump_to_pickle('text_sequences.pkl', text_seq)
# dump_to_pickle('vocab.pkl',vocab)
#
# id2w, w2id = convert_to_ids(text_seq,vocab)
#
# final_text2ids = convert_seq2ids(text_seq,id2w,w2id)
#
# dump_to_pickle('final_text2ids.pkl',final_text2ids)
# dump_to_pickle('id2w.pkl',id2w)
# dump_to_pickle('w2id',w2id)

# vocab = read_from_pickle('vocab.pkl')
# text_seq = read_from_pickle('text_sequences.pkl')
#seq2ids = read_from_pickle('final_text2ids.pkl')



def newSeq2id(sent_seq):
    try:
        new_seq = tokenize_sequence(sent_seq)
        return convert_seq2ids(new_seq,id2w,w2id)

    except ValueError as e:
        print(e)










