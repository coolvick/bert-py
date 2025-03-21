import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import nltk
from nltk.tokenize import word_tokenize
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout

nltk.download('punkt')
# nltk.download('punkt_tab')

# Load sample text dataset (replace with your own)
text = """
Deep learning is a subset of machine learning. It is inspired by the structure of the human brain.
Neural networks form the core of deep learning.
"""

# Tokenize wordspip
tokens = word_tokenize(text.lower())

# Create input-output pairs
seq_length = 5  # Number of words as input
input_sequences, next_words = [], []

for i in range(len(tokens) - seq_length):
    input_sequences.append(tokens[i:i + seq_length])
    next_words.append(tokens[i + seq_length])

# Build vocabulary
word_index = {word: i+1 for i, word in enumerate(set(tokens))}
index_word = {i: word for word, i in word_index.items()}

# Convert words to numerical sequences
input_sequences = [[word_index[word] for word in seq] for seq in input_sequences]
next_words = [word_index[word] for word in next_words]

# Pad sequences
input_sequences = pad_sequences(input_sequences, padding='post')

# Convert outputs to categorical format
output_labels = to_categorical(next_words, num_classes=len(word_index) + 1)
print('done');

# Load Universal Sentence Encoder
use_model = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")

# Convert input sequences to sentence embeddings
def get_use_embeddings(sentences):
    return use_model([" ".join([index_word[idx] for idx in seq if idx in index_word]) for seq in sentences]).numpy()

X_use = get_use_embeddings(input_sequences);

# Define LSTM model
model = Sequential([
    LSTM(128, return_sequences=True, input_shape=(X_use.shape[1], 1)),
    Dropout(0.2),
    LSTM(128),
    Dense(128, activation='relu'),
    Dense(len(word_index) + 1, activation='softmax')  # Output layer
])

# Compile model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train model
model.fit(X_use, output_labels, epochs=50, batch_size=16)

model.export('./tmp/next_word_model')
# print([x.name for x in model.signatures["serving_default"].outputs])
for layer in model.layers: print(layer.name)

print(dir(tf.compat))

# Convert to TensorFlow.js format
# !pip install tensorflowjs
# !tensorflowjs_converter --input_format=tf_saved_model --output_node_names='dense_1/Softmax' next_word_model tfjs_model