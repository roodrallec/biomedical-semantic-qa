import json
import re
import pandas as pd
import numpy as np
import pickle
import os.path
from gensim.models import Word2Vec
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.callbacks import EarlyStopping, TensorBoard
from keras.optimizers import Adam
from keras.layers.normalization import BatchNormalization
from keras.layers import Dense, LSTM, Embedding, Dropout, SpatialDropout1D, Bidirectional, Conv1D, MaxPooling1D
from nltk.stem import SnowballStemmer
from nltk.corpus import stopwords
from sklearn.metrics import classification_report
from matplotlib import pyplot as plt

TRAINING_DATA_DIR = './datasets/BioASQ-trainingDataset6b.json'
STEM = False  # SnowballStemmer('english')
ALLOWED_STOPWORDS = ['does', 'what', 'why', 'how', 'which', 'where', 'when', 'who']
WORD2VEC_PARAMS = {
    'size': 100,
    'min_count': 5,
    'sg': 0,  # 1 for skip gram 0 for bow
    'negative': 5,
    'window': 10,
    'workers': 16
}
# Training params
TEST_SPLIT = 0.1
VALIDATION_SPLIT = 0.1
USE_W2V_EMBED = False
TRAINABLE_EMBEDDING = False
LEARNING_RATE = 0.001
BATCH_SIZE = 128
DROPOUT = 0.3
EPOCHS = 100
# Logging
LOG_LEVEL = 0
RUN_NAME = "BATCH_SIZE: " + str(BATCH_SIZE) + \
           "; L_RATE: " + str(LEARNING_RATE) + \
           "; EMBED_VEC_SIZE: " + str(WORD2VEC_PARAMS['size']) + \
           "; DROPOUT: " + str(DROPOUT) + \
           "; W2V_EMBED: " + str(USE_W2V_EMBED)


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
            if STEM:
                word = STEM.stem(word)
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


def build_bi_lstm(embedding, output_size):
    model = Sequential()
    model.add(embedding)
    model.add(SpatialDropout1D(DROPOUT))
    model.add(Bidirectional(LSTM(WORD2VEC_PARAMS['size'])))
    model.add(Dropout(DROPOUT))
    model.add(BatchNormalization())
    model.add(Dense(output_size, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer=Adam(lr=LEARNING_RATE))
    return model


def build_conv_lstm(embedding, output_size):
    model = Sequential()
    model.add(embedding)
    model.add(Dropout(DROPOUT))
    model.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Bidirectional(LSTM(WORD2VEC_PARAMS['size'])))
    model.add(Dense(output_size, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer=Adam(lr=LEARNING_RATE))
    return model


def one_hot_encode_labels(q_labels, q_classes):
    y = []

    for q_label in q_labels:
        enc = np.zeros(len(q_classes))
        enc[q_classes.index(q_label)] = 1
        y.append(enc)
    return np.array(y)


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


def build_model(q_texts_tokens, vocab_idx_dict, max_seq_length, output_size):
    vocab_size = len(vocab_idx_dict.keys())
    vector_model = Word2Vec(q_texts_tokens, **WORD2VEC_PARAMS)
    embedding_weights = (np.random.rand(vocab_size, WORD2VEC_PARAMS['size']) - 0.5) / 5.0

    if USE_W2V_EMBED:
        for word, idx in vocab_idx_dict.items():
            try:
                embedding_weights[idx] = vector_model.wv[word]
            except KeyError:
                pass

    embedding = embedding_layer(input_sequence_size=max_seq_length,
                                embedding_weights=embedding_weights,
                                vocab_size=vocab_size)
    model = build_conv_lstm(embedding, output_size)
    return model


def embedding_layer(vocab_size, input_sequence_size, embedding_weights):
    embedding_arguments = {
        'input_dim': vocab_size,
        'input_length': input_sequence_size,
        'mask_zero': False,
        'output_dim': WORD2VEC_PARAMS['size'],
        'trainable': TRAINABLE_EMBEDDING,
        'weights': [embedding_weights]
    }

    return Embedding(**embedding_arguments)


def main():
    # Question pre-processing
    [q_texts, q_labels] = parse_questions(json_to_df(TRAINING_DATA_DIR))
    q_classes = list(np.unique(q_labels))
    q_texts_tokens = [text_to_tokens(text) for text in q_texts]

    # Embedding creation
    vocab_idx_dict = build_vocab_idx(q_texts_tokens)
    sequences = [[vocab_idx_dict[q_token] for q_token in q_tokens] for q_tokens in q_texts_tokens]
    max_seq_length = max([len(sequence) for sequence in sequences])

    model = build_model(q_texts_tokens, vocab_idx_dict, max_seq_length, len(q_classes))

    # Train test split
    y = one_hot_encode_labels(q_labels, q_classes)
    x = pad_sequences(sequences, maxlen=max_seq_length, padding="pre", truncating="post")
    train_idx = round(len(x) * (1 - TEST_SPLIT))
    x_train, y_train = x[:train_idx], y[:train_idx]
    x_test, y_test = x[train_idx:], y[train_idx:]

    # Model creation
    callbacks = [
        EarlyStopping(monitor='val_loss', min_delta=0, patience=20, verbose=0, mode='auto'),
        TensorBoard(log_dir="./Graph/"+RUN_NAME, histogram_freq=0, write_graph=True, write_images=True)
    ]
    hist = model.fit(x_train, y_train, callbacks=callbacks, validation_split=VALIDATION_SPLIT, epochs=EPOCHS,
                     batch_size=BATCH_SIZE, shuffle=True)

    history = pd.DataFrame(hist.history)
    plot_history(history)
    y_hat = model.predict(x_test)
    report = classification_report(np.argmax(y_test, axis=1), np.argmax(y_hat, axis=1), target_names=q_classes)
    logger(report, 1)
    return model


if __name__ == "__main__":
    main()
