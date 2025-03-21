import tensorflow as tf
import numpy as np
pad_sequences = tf.keras.preprocessing.sequence.pad_sequences
imdb = tf.keras.datasets.imdb


word_index = imdb.get_word_index(); #maps words to numerical indexes
reverse_word_index = {v: k for k,v in word_index.items()}; #reverse mapping

glove_path = "../data/glove.6B.100d.txt";
embeddings_index = {}

with open(glove_path, encoding="utf-8") as f:
    for line in f:
        values = line.split()
        word = values[0]
        vector = np.asarray(values[1:], dtype="float32") # rest are the embeddings
        embeddings_index[word] = vector

vocab_size = 10000
missing_words = 0;

for word, index in word_index.items():
    if index < vocab_size:
        if word not in embeddings_index:
            missing_words += 1

print(f"Missing words in Glove: {missing_words}/{vocab_size} ({(missing_words/vocab_size)*100:.2f}%)")