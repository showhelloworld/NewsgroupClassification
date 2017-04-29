from __future__ import print_function

import os
import numpy as np

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical
from keras.layers import Dense, Input, Flatten
from keras.layers import Conv1D, MaxPooling1D, Embedding
from keras.models import Model


np.random.seed(1010)

base_dir = './data'
glove_dir = base_dir + '/glove'
text_dir = base_dir + '/newsgroup'
max_seq_len = 1000
max_nb_words = 20000
embedding_dim = 100
validation_split = 0.2


print('Indexing word vectors...')
embedding_index = {}
with open(os.path.join(glove_dir, 'glove.6B.100d.txt')) as f:
    for line in f:
        parts = line.split()
        word = parts[0]
        coeffs = np.asarray(parts[1:], dtype='float32')
        embedding_index[word] = coeffs
print('Word vectors found:', len(embedding_index))

print('Processing text dataset')
texts = []
labels_index = {}
labels = []
label_id = 0
for name in sorted(os.listdir(text_dir)):
    path = os.path.join(text_dir, name)
    if os.path.isdir(path):
        labels_index[name] = label_id
        for fname in sorted(os.listdir(path)):
            if fname.isdigit():
                fpath = os.path.join(path, fname)
                with open(fpath, encoding='latin-1') as f:
                    texts.append(f.read())
                labels.append(label_id)
        label_id += 1
print('Texts found:', len(texts))

tokenizer = Tokenizer(nb_words=max_nb_words)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)
word_index = tokenizer.word_index
print('Unique tokens:', len(word_index))
data = pad_sequences(sequences, maxlen=max_seq_len)
labels = to_categorical(np.asarray(labels))
print('Shape of data:', data.shape)
print('Shape of labels:', labels.shape)

indices = np.arange(data.shape[0])
np.random.shuffle(indices)
data = data[indices]
labels = labels[indices]
nb_validation_samples = int(validation_split*data.shape[0])
x_train = data[:-nb_validation_samples]
y_train = labels[:-nb_validation_samples]
x_valid = data[-nb_validation_samples:]
y_valid = labels[-nb_validation_samples:]
print('Data split')

nb_words = min(max_nb_words, len(word_index))
embedding_matrix = np.zeros((nb_words+1, embedding_dim))
for word, i in word_index.items():
    if i > max_nb_words:
        continue
    embedding_vector = embedding_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector
print('Embedding matrix prepared.')

embedding_layer = Embedding(nb_words+1, embedding_dim, weights=[embedding_matrix],
        input_length=max_seq_len, trainable=False)

print('Training model')
sequence_input = Input(shape=(max_seq_len,), dtype='int32')
embedding_sequences = embedding_layer(sequence_input)
x = Conv1D(128, 5, activation='relu')(embedding_sequences)
x = MaxPooling1D(5)(x)
x = Conv1D(128, 5, activation='relu')(x)
x = MaxPooling1D(5)(x)
x = Conv1D(128, 5, activation='relu')(x)
x = MaxPooling1D(5)(x)
x = Flatten()(x)
x = Dense(128, activation='relu')(x)
preds = Dense(len(labels_index), activation='softmax')(x)

model = Model(sequence_input, preds)
model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['acc'])
model.fit(x_train, y_train, validation_data=(x_valid, y_valid), nb_epoch=2, batch_size=128)


