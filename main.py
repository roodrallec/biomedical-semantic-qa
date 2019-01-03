

###
# DEPENDENCIES
###


import json
import re
import pandas as pd
import numpy as np
import pickle
import os.path
from gensim.models import Word2Vec
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, LSTM, Embedding, Dropout, SpatialDropout1D, Bidirectional
from keras.models import Model
from keras.callbacks import EarlyStopping, TensorBoard
from keras.optimizers import Adam
from keras.layers.normalization import BatchNormalization
from nltk.stem import SnowballStemmer
from nltk.corpus import stopwords
from sklearn.metrics import classification_report
from matplotlib import pyplot as plt


###
# GLOBALS
###


TRAINING_DATA_DIR = './datasets/BioASQ-trainingDataset6b.json'
TEST_SPLIT = 0.1
VALIDATION_SPLIT = 0.1
STEM = SnowballStemmer('english')
USE_W2V_EMBED = False
USE_W2V_SKIP_GRAM = 1  # 1 for skip gram 0 for bow
W2V_MIN_COUNT = 0
EMBEDDING_VECTOR_SIZE = 90
LEARNING_RATE = 0.001
EPOCHS = 200
BATCH_SIZE = 128
DROPOUT = 0.3
LOG_LEVEL = 0
ALLOWED_STOPWORDS = ['does', 'what', 'why', 'how', 'which', 'where', 'when', 'who']
RUN_NAME = "BATCH_SIZE: " + str(BATCH_SIZE) + \
           "; L_RATE: " + str(LEARNING_RATE) + \
           "; EMBED_VEC_SIZE: " + str(EMBEDDING_VECTOR_SIZE) + \
           "; DROPOUT: " + str(DROPOUT) + \
           "; W2V_EMBED: " + str(USE_W2V_EMBED)

###
# FUNCTIONS
###


def logger(text, level=0):
    if level >= LOG_LEVEL:
        print(text)


def parse_questions(data):
    return zip(*[[d['body'], d['type']] for d in data['questions']])


def label_to_class(str_labels, label):
    return str_labels.index(label)


def json_to_df(json_file_path):
    with open(json_file_path, 'r') as f:
        return pd.DataFrame(json.load(f))


def text_to_tokens(text):
    logger(text)
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
    ignore_words = set(stopwords.words("english")) - set(ALLOWED_STOPWORDS)

    for word in text.split():
        if word not in ignore_words:
            tokens.append(word)

    logger(tokens)
    return tokens


def build_vocab_idx(questions_tokens):
    vocab = dict()
    idx = 0
    for question_tokens in questions_tokens:
        for token in question_tokens:
            if token not in vocab:
                vocab[token] = idx
                idx += 1
    return vocab


def build_bi_lstm(input_sequence_size, embedding_weights, vocab_size, output_size):
    # Inputs
    sequence_input = Input(shape=(input_sequence_size,), dtype='int32')
    # Embedding layer
    embedding_arguments = {
        'input_dim': vocab_size,
        'output_dim': EMBEDDING_VECTOR_SIZE,
        'mask_zero': False,
        'input_length': input_sequence_size,
        'trainable': False
    }

    if USE_W2V_EMBED:
        embedding_arguments['weights'] = [embedding_weights]

    x = Embedding(**embedding_arguments)(sequence_input)
    x = SpatialDropout1D(DROPOUT)(x)
    # Bi-directional LSTM
    x = Bidirectional(LSTM(EMBEDDING_VECTOR_SIZE, return_sequences=False))(x)
    x = Dropout(DROPOUT)(x)
    x = BatchNormalization()(x)
    # Output prediction layer
    predictions = Dense(output_size, activation='sigmoid')(x)
    # Model compilation
    model = Model(inputs=sequence_input, outputs=predictions)
    model.compile(loss='binary_crossentropy', optimizer=Adam(lr=LEARNING_RATE), metrics=[])
    return model


def one_hot_encode_labels(q_labels, q_classes):
    y = []

    for q_label in q_labels:
        enc = np.zeros(len(q_classes))
        enc[q_classes.index(q_label)] = 1
        y.append(enc)
    return np.array(y)


def train_test_split(q_labels, q_classes, sequences, max_seq_length):
    y = one_hot_encode_labels(q_labels, q_classes)
    x = pad_sequences(sequences, maxlen=max_seq_length, padding="pre", truncating="post")
    train_idx = round(len(x)*(1-TEST_SPLIT))
    x_train, y_train = x[:train_idx], y[:train_idx]
    x_test, y_test = x[train_idx:], y[train_idx:]
    return x_train, y_train, x_test, y_test


def plot_history(history):
    title = RUN_NAME.split('; ')
    plt_title = '; '.join(title[:3] + ["\n"] + title[3:])
    plt.figure(figsize=(5, 5))
    plt.plot(history["loss"], 'g', label="training")
    plt.plot(history["val_loss"], 'r', label="validation")
    plt.yticks(np.arange(0, 1, 0.05))
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend(loc='upper right')
    plt.title(plt_title)
    plt.show()


def main():
    # Question pre-processing
    [q_texts, q_labels] = parse_questions(json_to_df(TRAINING_DATA_DIR))
    q_classes = list(np.unique(q_labels))

    q_tokens_pickle = "q_tokens.p"
    if os.path.isfile(q_tokens_pickle):
        q_texts_tokens = pickle.load(open(q_tokens_pickle, "rb"))
    else:
        q_texts_tokens = [text_to_tokens(text) for text in q_texts]
        pickle.dump(q_texts_tokens, open(q_tokens_pickle, "wb"))

    max_seq_length = max([len(q_tokens) for q_tokens in q_texts_tokens])
    vocab_idx = build_vocab_idx(q_texts_tokens)
    sequences = [[vocab_idx[q_token] for q_token in q_tokens] for q_tokens in q_texts_tokens]
    vector_model = Word2Vec(q_texts_tokens,
                            min_count=W2V_MIN_COUNT,
                            sg=USE_W2V_SKIP_GRAM,
                            size=EMBEDDING_VECTOR_SIZE)
    vocab_size = len(vector_model.wv.vocab)
    embedding_weights = (np.random.rand(vocab_size, EMBEDDING_VECTOR_SIZE) - 0.5) / 5.0

    for word, i in vocab_idx.items():
        embedding_weights[i] = vector_model.wv[word]

    x_train, y_train, x_test, y_test = train_test_split(q_labels, q_classes, sequences, max_seq_length)

    model = build_bi_lstm(
        input_sequence_size=max_seq_length,
        embedding_weights=embedding_weights,
        vocab_size=vocab_size,
        output_size=len(q_classes)
    )
    callbacks = [
        EarlyStopping(monitor='val_loss', min_delta=0, patience=20, verbose=0, mode='auto'),
        TensorBoard(log_dir="./Graph/"+RUN_NAME, histogram_freq=0, write_graph=True, write_images=True)
    ]
    hist = model.fit(x_train, y_train,
                     callbacks=callbacks, validation_split=VALIDATION_SPLIT,
                     epochs=EPOCHS, batch_size=BATCH_SIZE, shuffle=True)

    history = pd.DataFrame(hist.history)
    plot_history(history)
    # RESULTS
    y_hat = model.predict(x_test)
    logger(classification_report(np.argmax(y_test, axis=1),
                                 np.argmax(y_hat, axis=1),
                                 target_names=q_classes), 1)


if __name__ == "__main__":
    main()
