###
# DEPENDENCIES
###
import nltk
import json
import re
import pandas as pd
import numpy as np
from string import punctuation
from gensim.models import Word2Vec
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, LSTM, Embedding, Dropout,SpatialDropout1D, Bidirectional
from keras.models import Model
from keras.optimizers import Adam
from keras.layers.normalization import BatchNormalization
from nltk.stem import SnowballStemmer
from nltk.corpus import stopwords
from sklearn.metrics import classification_report
from matplotlib import pyplot as plt
###
# GLOBALS
###
TRAINING_DATA_DIR="./datasets/BioASQ-trainingDataset6b.json"
TRAIN_SPLIT = 0.9
W2V_SIZE=100
W2V_SKIP_GRAM=1
W2V_MIN_COUNT=0
STEMMER = SnowballStemmer('english')
###
# FUNCTIONS
###
def parse_questions_types(data):
    return zip(*[[json['body'], json['type']] for json in data['questions']])


def label_to_class(str_labels, label):
    return str_labels.index(label)


def json_to_df(json_file_path):
    with open(json_file_path, 'r') as f:
        return pd.DataFrame(json.load(f))


def text_to_tokens(text):
    print(text)
    text = text.lower()
    text = re.sub(r"\?", " question ", text)
    text = re.sub("\d+", " digit ", text)
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\'s", " is ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"i'm", " i am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)

    tokens = []
    for word in text.split():
        if word not in ['what', 'why', 'how', 'which', 'where', 'when', 'who']:
            if word in set(stopwords.words("english")):
                continue
        tokens.append(word) # STEMMER.stem(word)

    print(tokens)
    return tokens

def build_vocab_idx(tokenized_q):
    vocab = dict()
    idx = 0
    for q in tokenized_q:
        for word in q:
            if word not in vocab:
                vocab[word] = idx
                idx += 1
    return vocab


def one_hot_encode(q_types):
    y = []
    list_classes = list(np.unique(q_types))
    for q_type in q_types:
        enc = np.zeros(len(list_classes))
        enc[list_classes.index(q_type)] = 1
        y.append(enc)
    return np.array(y)

[questions, q_types] = parse_questions_types(json_to_df(TRAINING_DATA_DIR))
tokenized_q = [text_to_tokens(q) for q in questions]
vocab_idx = build_vocab_idx(tokenized_q)
vector_model = Word2Vec(tokenized_q, min_count=W2V_MIN_COUNT, sg=W2V_SKIP_GRAM, size=W2V_SIZE)
max_nb_words = len(vector_model.wv.vocab)
max_seq_length = max([len(q) for q in tokenized_q])
sequences = [[vocab_idx[t] for t in q] for q in tokenized_q]


X = pad_sequences(sequences, maxlen=max_seq_length, padding="pre", truncating="post")
y = one_hot_encode(q_types)
train_q_idx = round(len(X)*TRAIN_SPLIT)
X_train, y_train = X[:train_q_idx], y[:train_q_idx]
X_test, y_test = X[train_q_idx:], y[train_q_idx:]


wv_matrix = (np.random.rand(max_nb_words, W2V_SIZE) - 0.5) / 5.0

for word, i in vocab_idx.items():
    wv_matrix[i] = vector_model.wv[word]

wv_layer = Embedding(max_nb_words,
                     W2V_SIZE,
                     mask_zero=False,
                     weights=[wv_matrix],
                     input_length=max_seq_length,
                     trainable=False)

# Inputs
sequence_input = Input(shape=(max_seq_length,), dtype='int32')
embedded_sequences = wv_layer(sequence_input)

# biGRU
embedded_sequences = SpatialDropout1D(0.2)(embedded_sequences)
x = Bidirectional(LSTM(W2V_SIZE, return_sequences=False))(embedded_sequences)

# Output
x = Dropout(0.2)(x)
x = BatchNormalization()(x)
output = Dense(4, activation='sigmoid')(x)

# build the model
model = Model(inputs=[sequence_input], outputs=output)
model.compile(loss='binary_crossentropy',
              optimizer=Adam(lr=0.002),
              metrics=[])

hist = model.fit(X_train, y_train, validation_split=0.1, epochs=30,
                 batch_size=128, shuffle=True)

history = pd.DataFrame(hist.history)
plt.figure(figsize=(5, 5))
plt.plot(history["loss"], 'g')
plt.plot(history["val_loss"], 'r')
plt.title("Loss with pretrained word vectors")
plt.show()

y_hat = model.predict(X_test)
print(classification_report(np.argmax(y_test, axis=1),
                            np.argmax(y_hat, axis=1),
                            target_names=np.unique(q_types)))