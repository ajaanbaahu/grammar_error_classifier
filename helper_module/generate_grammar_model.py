#!/Users/rmohan/flaskapp/lib/python3.4

import sys
sys.path.insert(0, 'grammar-error-check')

from grammar_models.model_seq import read_data, simple_LSTM

import tensorflow as tf


tf.app.flags.DEFINE_string("input_data",None,"path to training data")
tf.app.flags.DEFINE_integer("train_samples",1000000,'samples to train')
tf.app.flags.DEFINE_string("test_samples",500000,'sampels to test')
tf.app.flags.DEFINE_integer("max_seq_len",100,'size of data point')
tf.app.flags.DEFINE_integer("size", 100, "Size of each model layer.")
tf.app.flags.DEFINE_integer("vocab_size",300000,"size of vocabulary")
tf.app.flags.DEFINE_integer("batch_size",1024,"size of invidiual batches")
tf.app.flags.DEFINE_integer("embedding_size",64,"embed vector len")

FLAGS = tf.app.flags.FLAGS

source_file, train_samples, test_sample,max_review_length = FLAGS.input_data, FLAGS.train_samples, FLAGS.test_samples,FLAGS.max_seq_len


X_train,y_train,X_test,y_test = read_data(source_file, train_samples, test_sample,max_review_length)

embedding_vector_length,max_review_length,num_hidden_units = FLAGS.embedding_size, FLAGS.max_seq_len, FLAGS.size

model = simple_LSTM(embedding_vector_length,max_review_length,num_hidden_units)
#model = LSTM_dropput()
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=5, batch_size=1024)

scores = model.evaluate(X_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))

# serialize model to JSON
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model.h5")
print("Saved model to disk")


# print(len(X),len(y),len(x))