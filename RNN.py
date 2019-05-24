from collections import defaultdict
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.layers import LSTM
from keras.optimizers import RMSprop
from keras.utils.data_utils import get_file
from keras import backend as K
import random
import sys
import pandas as pd

data = pd.read_csv("DATA1.txt", header=None, delimiter='\t', encoding='iso-8859-1')
text = ''
for i in range(len(data)):
    text = text + data[1][i]

chars = sorted(list(set(text))) # getting all unique chars
char_len = len(chars)

print('----------------')
for i in range(len(data)):
    print(i)
    text = data[1][i]
    #print('text length', len(text))
    #chars = sorted(list(set(text))) # getting all unique chars
    #char_len = len(chars)
    #print('total chars: ', len(chars))
    char_indices = dict((c, i) for i, c in enumerate(chars))
    indices_char = dict((i, c) for i, c in enumerate(chars))
    emotion = data[0][i]
    # Generates character sequences
    maxlen = 40
    step = 1
    sentences = []
    next_chars = []
    for i in range(0, len(text) - maxlen, step):
        sentences.append(text[i: i + maxlen])
        next_chars.append(text[i + maxlen])
    print('nb sequences:', len(sentences))
    if (len(sentences) == 0):
        continue;
    # Puts sequences into X and Y matrices to be trained
    print('Vectorization...')
    print("MAX LEN IS " + str(maxlen))
    print("LEN CHARS IS " + str(len(chars)))
    x = np.zeros((len(sentences), maxlen, len(chars)), dtype=np.bool)
    y = np.zeros((len(sentences), len(chars)), dtype=np.bool)

    print("hi")
    print(x)
    for i, sentence in enumerate(sentences):
        for t, char in enumerate(sentence):
            x[i, t, char_indices[char]] = 1
        y[i, char_indices[next_chars[i]]] = 1

    print("oof")
    print(x)


    # Trains model
    model = Sequential()
    model.add(LSTM(200, input_shape=(maxlen, char_len)))

    model.add(Dense(char_len))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

    print("hey got here")
    print(x)

    model_history = model.fit(x, y,
              batch_size=50,
              epochs=1)

    inp = model.input
    outputs = [layer.output for layer in model.layers]
    functors = [K.function([inp, K.learning_phase()], [out]) for out in outputs]

    # Testing
    test = np.random.random((40, char_len))[np.newaxis,...]
    layer_outs = [func([test, 1.]) for func in functors]
    last = layer_outs[len(layer_outs) - 1]

    f = open("layers.txt", "a")
    f.write('[')
    for i in range(len(last[0][0])):
        f.write(str(last[0][0][i]) + ', ')
    f.write(emotion + '],\n')
    f.close()
