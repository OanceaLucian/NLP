import pickle
import re

import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical

from keras.layers import Embedding
from keras.layers import Dense, Input, Flatten
from keras.layers import Conv1D, MaxPooling1D, Embedding, Dropout
from keras.models import Model
from keras.layers.merge import Concatenate
from keras.callbacks import ModelCheckpoint, Callback, EarlyStopping
from gensim.models import word2vec
from gensim.models.keyedvectors import KeyedVectors
MAX_SEQUENCE_LENGTH = 100
MAX_NB_WORDS = 20000
EMBEDDING_DIM = 300
VALIDATION_SPLIT = 0.1

import tensorflow as tf
tf.set_random_seed(666)
np.random.seed(666)

def clean_str(string):
    """
    Tokenization/string cleaning for dataset
    Every dataset is lower cased except
    """
    string = re.sub(r"\\", "", string)
    string = re.sub(r"\'", "", string)
    string = re.sub(r"\"", "", string)
    return string.strip().lower()

with open("non_humorous.txt","r") as f:
    wikipedia = f.readlines()

with open("humorous.txt","r") as f:
    one_liners = f.readlines()
text_train = []
label_train = []
text_test = []
label_test = []
full_text = one_liners+wikipedia
for x in range(0,int(2*len(one_liners)/3)):
   # print x,one_liners[x]
    text_train.append(clean_str(one_liners[x]))
    label_train.append(1)
for x in range(0,int(2*len(one_liners)/3)):
    #print x,wikipedia[x]
    text_train.append(clean_str(wikipedia[x]))
    label_train.append(0)

for x in range(int(2*len(one_liners)/3+1),int(len(one_liners))):
    text_test.append(clean_str(one_liners[x]))
    label_test.append(1)
for x in range(int(2*len(one_liners)/3+1),int(len(one_liners))):
    text_test.append(clean_str(wikipedia[x]))
    label_test.append(0)

tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
tokenizer.fit_on_texts(full_text)
sequences = tokenizer.texts_to_sequences(text_train)
data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)

test_sequences = tokenizer.texts_to_sequences(text_test)
data_test = pad_sequences(test_sequences, maxlen=MAX_SEQUENCE_LENGTH)
labels = to_categorical(np.asarray(label_train))
test_labels = to_categorical(np.asarray(label_test))
print('Shape of data tensor:', data.shape)
print('Shape of label tensor:', labels.shape)

print('Shape of data tensor:', data_test.shape)
print('Shape of label tensor:', test_labels.shape)

word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))

indices = np.arange(data.shape[0])
np.random.shuffle(indices)
data = data[indices]
labels = labels[indices]
nb_validation_samples = int(VALIDATION_SPLIT * data.shape[0])

x_train = data[:-nb_validation_samples]
y_train = labels[:-nb_validation_samples]
x_val = data[-nb_validation_samples:]
y_val = labels[-nb_validation_samples:]

print('Number of positive and negative reviews in traing and validation set ')
print( y_train.sum(axis=0))
print( y_val.sum(axis=0))
load_model = 1
if load_model== 1:
    model_name = "GoogleNews-vectors-negative300.bin"
    embedding_matrix = np.random.random((len(word_index) + 1, EMBEDDING_DIM))
    embedding_model = KeyedVectors .load_word2vec_format(model_name, binary=True)
    for word, i in word_index.items():
        if word in embedding_model:
            embedding_vector = embedding_model[word]
        else:
            embedding_vector = None
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector

    embedding_layer = Embedding(len(word_index) + 1,
                            EMBEDDING_DIM,
                            weights=[embedding_matrix],
                            input_length=MAX_SEQUENCE_LENGTH,
                            trainable=True)
    sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
    embedded_sequences = embedding_layer(sequence_input)
    embedded_sequences = Dropout(0.7)(embedded_sequences)
    convs = []
    filter_sizes = [5,6,7]
    filter_length = 128
    for fsz in filter_sizes:
        l_conv = Conv1D(filters=filter_length,kernel_size=fsz,activation='relu')(embedded_sequences)
        l_pool = MaxPooling1D(5)(l_conv)
        convs.append(l_pool)
    l_merge = Concatenate(axis=1)(convs)
    l_flat = Flatten()(l_merge)
    l_dense = Dense(128, activation='relu')(l_flat)
    l_dense= Dropout(0.35)(l_dense)
    preds = Dense(2, activation='softmax')(l_dense)

    model = Model(sequence_input, preds)

train = 0
if train == 1:
    model.compile(loss='categorical_crossentropy',
                  optimizer='rmsprop',
                  metrics=['acc'])

    print("model fitting - simplified convolutional neural network")
    model.summary()
    filepath = "ted_all_weights-improvement_50_-{epoch:02d}-{val_acc:.2f}.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    earlyStopp = EarlyStopping(monitor='val_acc', patience=10, verbose=0, mode='auto')
    callbacks_list = [checkpoint, earlyStopp]
    model.fit(x_train, y_train, validation_data=(x_val, y_val),
          nb_epoch=50, batch_size=128, verbose = 1,callbacks=callbacks_list)
predict = 1
if predict:
    model.load_weights("ted_all_weights-improvement_50_-09-0.60.hdf5")
    model.compile(loss='categorical_crossentropy',
                  optimizer='rmsprop',
                  metrics=['acc'])

    print("model predict - simplified convolutional neural network")
    model.summary()
    result = model.evaluate(data_test,test_labels,batch_size = 32,verbose = 1)
    for x in range(0,len(result)):
        print("%s: %.2f%%" % (model.metrics_names[x], result[x]))
    print(result)
