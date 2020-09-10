# Reference: https://cirocavani.github.io/post/tensorflow-word-embedding-com-word2vec/

import collections
import os
import tensorflow as tf
import random
import zipfile
from scipy.spatial import distance_matrix
import pandas as pd
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np


# Settings and Hyperparameters
vocabulary_size = 5000
EMBEDDING_DIM = 2
BATCH_SIZE = 50
WINDOW_SIZE = 5
EPOCHS = 2

# Loading wikipedia data -----------------------------------------------------------------------------------------------
# In Memory read
# from io import BytesIO
# from zipfile import ZipFile
# from urllib.request import urlopen
#
# resp = urlopen('http://mattmahoney.net/dc/text8.zip')
# zipfile = ZipFile(BytesIO(resp.read()))
# zipfile.namelist()
# with zipfile.ZipFile('text8') as f:
#     raw_text = f.read(f.namelist()[0]).decode('utf-8')
with zipfile.ZipFile(os.path.join('.', 'data', 'text8.zip')) as f:
    raw_text = f.read(f.namelist()[0]).decode('utf-8')

words = raw_text.split()


# Preprocessing --------------------------------------------------------------------------------------------------------
# words_freq = collections.Counter(words).most_common()
#
# words_vocab = words_freq[:(vocabulary_size-1)]  # get 50000 most frequent words
#
# # Defining token for unknown words
# UNK_ID = 0
# word_to_id = dict((word, word_id)
#                   for word_id, (word, _) in enumerate(words_vocab, UNK_ID+1))
# word_to_id['UNK'] = UNK_ID
# word_from_id = dict((word_id, word) for word, word_id in word_to_id.items())
#
# with open('vocabulary.txt', 'w') as f:
#     for word_id in range(vocabulary_size):
#         f.write(word_from_id[word_id] + '\n')

# Obtain necessary variables directly
with open('vocabulary.txt', 'r') as f:
    vocab = f.read().split('\n')[:-1]

UNK_ID = 0
word_to_id = dict((word, word_id)
                  for word_id, word in enumerate(vocab[:vocabulary_size]))
word_from_id = dict((word_id, word) for word, word_id in word_to_id.items())

data = list(word_to_id.get(word, UNK_ID) for word in words)  # transforming corpus to index format

# Continuous Bag-of-Words (CBOW) ---------------------------------------------------------------------------------------
# Input function


def context_window(window_words, target_index):
    words = list(window_words)
    del words[target_index]
    return words


def cbow_input(data, batch_size, window_size):
    """
    Data Generator that yields batches of (context, center) pairs, where context are the context words within a
    window_size and center is the center word of the window_size range.
    :param data: list of words indexes that represent the training corpus.
    :param batch_size: size of each batch to be passed to the model.
    :param window_size: defines the amount of context used to predict the center word. Should be a odd number greater
    than 3.
    :return: yields ((batch_size, window_size - 1), (batch_size, 1)) numpy arrays
    """
    if window_size % 2 == 0 or window_size < 3 or window_size > (len(data) - batch_size) / 2:
        # {window_size} must be odd: (n words left), target, (n words right)
        raise Exception(
            'Invalid parameters: window_size must be a small odd number')

    num_words = len(data)
    num_windows = num_words - window_size + 1
    num_batches = num_windows // batch_size
    target_index = window_size // 2

    words = collections.deque(data[window_size:])  # Deques have O(1) speed for appendleft() and popleft() while lists
    # have O(n) performance for insert(0, value) and pop(0). Deques data structures are a generalization of stacks and queues.
    window_words = collections.deque(data[:window_size], maxlen=window_size)

    # Build each batch
    for n in range(num_batches):
        batch = np.ndarray(shape=(batch_size, window_size - 1), dtype=np.int32)
        labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)

        # Build each example in a batch
        for m in range(batch_size):
            batch[m, :] = context_window(window_words, target_index)  # build X
            labels[m, 0] = window_words[target_index]  # build y
            window_words.append(words.popleft())  # add the removed leftmost element of words to window_words

        yield batch, labels


# # Example
# batch_size = 2
# window_size = 3
# num_iters = 2
# num_words = window_size + num_iters * batch_size - 1
# text = ' '.join(word_from_id[word_id] for word_id in data[:num_words])
# print('Text\n\n', text, '\n')
#
# data_iter = cbow_input(data, batch_size, window_size)
# for k in range(1, num_iters+1):
#     print('Batch {}\n'.format(k))
#     batch_context, batch_target = next(data_iter)
#     for i in range(batch_size):
#         context_words = ', '.join(
#             word_from_id[word_id] for word_id in batch_context[i, :])
#         target_word = word_from_id[batch_target[i, 0]]
#         print('[{}] -> {}'.format(context_words, target_word))
#     print()


# # Creation of datasets
# types = (tf.float32, tf.float32)
# shapes = ((WINDOW_SIZE - 1), (1,))
# ds_train = tf.data.Dataset.from_generator(lambda: cbow_input(), types, shapes).shuffle(1000).batch(BATCH_SIZE)

# Model
cbow_model = keras.Sequential([
    layers.Embedding(vocabulary_size, EMBEDDING_DIM),  # Used to map indices to embeddings
    layers.GlobalAveragePooling1D(),  # Average out embeddings for every context word in an example
    layers.Dense(vocabulary_size, activation='softmax')  # Output probability of each word being the center word
])

print(cbow_model.summary())

# Compile and train the model
cbow_model.compile(optimizer='adam', loss='categorical_crossentropy')

steps = len(data) // BATCH_SIZE  # Make such that each epoch covers all possible windows
cbow_model.fit(cbow_input(data, BATCH_SIZE, WINDOW_SIZE), epochs=EPOCHS, steps_per_epoch=steps)

# Get word embeddings
weights = cbow_model.get_weights()[0]
weights = weights[1:]  # removed 'UNK' token
print(weights.shape)

pd.DataFrame(weights, index=list(word_from_id.values())[1:]).head(8)

# Get distance matrix and view contextually similar words
euclid_matrix = distance_matrix(weights, weights)

similar_words = {search_term: [word_from_id[idx] for idx in euclid_matrix[word_to_id[search_term]-1].argsort()[1:4]+1]
                 for search_term in ['one', 'zero']}

print(similar_words)


# Skip-gram ------------------------------------------------------------------------------------------------------------
# Input function

def context_window(window_words, target_index):
    words = list(window_words)
    del words[target_index]
    return words


def context_sample(context_words, sample_size):
    return random.sample(context_words, sample_size)


def context_skips(window_words, target_index, sample_size, use_sample):
    words = context_window(window_words, target_index)
    if use_sample:
        words = context_sample(words, sample_size)
    return words


def skipgram_input(data, batch_size, window_size, num_skips):
    if window_size % 2 == 0 or window_size < 3 \
            or window_size > (len(data) - batch_size) / 2:
        # {window_size} must be odd: (n words left) target (n words right)
        raise Exception(
            'Invalid parameters: window_size must be a small odd number')
    if num_skips > window_size - 1:
        # It is not possible to generate {num_skips} different pairs
        # with the second word coming from {window_size - 1} words.
        raise Exception(
            'Invalid parameters: num_skips={}, window_size={}'.format(
                num_skips, window_size))

    num_words = len(data)
    num_windows = num_words - window_size + 1
    num_batches = num_windows * num_skips // batch_size
    target_index = window_size // 2
    use_sample = num_skips < window_size - 1

    words = collections.deque(data[window_size:])
    window_words = collections.deque(data[:window_size], maxlen=window_size)
    target_word = window_words[target_index]
    context_words = context_skips(window_words,
                                  target_index,
                                  num_skips,
                                  use_sample)

    for n in range(num_batches):
        batch = np.ndarray(shape=(batch_size,), dtype=np.int32)
        labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)

        for m in range(batch_size):
            batch[m] = target_word
            labels[m, 0] = context_words.pop()
            if not context_words:
                window_words.append(words.popleft())
                target_word = window_words[target_index]
                context_words = context_skips(window_words,
                                              target_index,
                                              num_skips,
                                              use_sample)
        yield batch, labels


# Example
batch_size = 2
window_size = 3
num_skips = 2
num_iters = 2
num_words = window_size + num_iters * batch_size // num_skips - 1
text = ' '.join(word_from_id[word_id] for word_id in data[:num_words])
print('Text\n\n', text, '\n')

data_iter = skipgram_input(data, batch_size, window_size, num_skips)
for k in range(1, num_iters+1):
    print('Batch {}\n'.format(k))
    batch_target, batch_context = next(data_iter)
    for i in range(batch_size):
        target_word = word_from_id[batch_target[i]]
        context_word = word_from_id[batch_context[i, 0]]
        print('{} -> {}'.format(target_word, context_word))
    print()

# Model
skipgram_model = keras.Sequential([
    layers.Embedding(vocabulary_size, EMBEDDING_DIM),  # Used to map indices to embeddings
    layers.Dense(vocabulary_size, activation='softmax')  # Output probability of each word being the center word
])

print(skipgram_model.summary())

# Compile and train the model
skipgram_model.compile(optimizer='adam', loss='categorical_crossentropy')

steps = len(data) // BATCH_SIZE  # Make such that each epoch covers all possible windows
cbow_model.fit(cbow_input(data, BATCH_SIZE, WINDOW_SIZE), epochs=EPOCHS, steps_per_epoch=steps)
