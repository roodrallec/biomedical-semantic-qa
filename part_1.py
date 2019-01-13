import json
import re
import pandas as pd
import numpy as np
import pickle
import os
from sklearn.model_selection import StratifiedKFold
from datetime import datetime
from gensim.models import Word2Vec
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.callbacks import EarlyStopping, TensorBoard
from keras.optimizers import Adam
from keras.layers.normalization import BatchNormalization
from keras.layers import Dense, LSTM, Embedding, SpatialDropout1D, Bidirectional, Conv1D, MaxPooling1D
from nltk.stem import SnowballStemmer
from nltk.corpus import stopwords
from sklearn.metrics import classification_report
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split

# RUN CONFIG
TRAINING_DATA_DIR = './datasets/BioASQ-trainingDataset6b.json'
QUORA_DATA_DIR = './datasets/quora_duplicate_questions.tsv'
STEM = False
Q_CLASSES = ['factoid', 'list', 'summary', 'yesno']
# ALLOWED_STOPWORDS = ['does', 'what', 'why', 'how', 'which', 'where', 'when', 'who', 'is', 'are']
TEST_SPLIT = 0.1
VALIDATION_SPLIT = 0.1
TRAINABLE_EMBEDDING = False
USE_CONV = True
CROSS_VALIDATION = False
LEARNING_RATE = 0.001
BATCH_SIZE = 128
DROPOUT = 0.3
EPOCHS = 100
USE_W2V_EMBED = True
WORD2VEC_PARAMS = {
    'size': 100,
    'min_count': 5,
    'sg': 0,  # 1 for skip gram 0 for bow
    'negative': 5,
    'window': 10,
    'workers': 16
}
# END RUN CONFIG

# Logging
LOG_LEVEL = 1
LOG_DIR = "./Graphs_datetime/" + str(datetime.utcnow())


def get_test_split():
    return TEST_SPLIT


def get_q_texts_tokens():
    # Question pre-processing
    [q_texts, q_labels] = parse_questions(json_to_df(TRAINING_DATA_DIR))
    q_texts_tokens = [text_to_tokens(text) for text in q_texts]
    return q_texts_tokens, q_labels


def parse_questions(data):
    return zip(*[[d['body'], d['type']] for d in data['questions']])


def label_to_class(str_labels, label):
    return str_labels.index(label)


def json_to_df(json_file_path):
    with open(json_file_path, 'r') as f:
        return pd.DataFrame(json.load(f))


def text_to_tokens(text):
    logger(text, 0)
    text = text.lower()
    text = re.sub(r"\?", " QUESTION ", text)
    text = re.sub("\d+", " DIGIT ", text)
    text = re.sub(r"\'s", " is ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"i'm", " i am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r"[^\w\s]", " ", text)
    tokens = []
    ignore_words = []
    # (set(stopwords.words("english")) - set(ALLOWED_STOPWORDS))

    for word in text.split():
        if word not in ignore_words:
            if STEM:
                word = SnowballStemmer('english').stem(word)
            tokens.append(word)

    logger(tokens, 0)
    return tokens


def build_vocab(tokens_list_list):
    vocab = dict()
    idx = 0
    vocab['_PADDING_'] = idx  # reserve idx 0 for padding
    idx += 1
    for tokens_list in tokens_list_list:
        for token in tokens_list:
            if token not in vocab:
                vocab[token] = idx
                idx += 1
    return vocab


def one_hot_encode_labels(q_labels):
    q_classes = get_q_classes()
    y = []

    for q_label in q_labels:
        enc = np.zeros(len(q_classes))
        enc[q_classes.index(q_label)] = 1
        y.append(enc)

    return np.array(y)


def build_model(vocab, max_seq_length, output_size, embed_w2v, dropout, learning_rate):
    vocab_tokens = vocab.keys()
    vocab_size = len(vocab_tokens)
    vector_model = Word2Vec(vocab_tokens, **WORD2VEC_PARAMS)
    embedding_weights = (np.random.rand(vocab_size, WORD2VEC_PARAMS['size']) - 0.5) / 5.0

    if embed_w2v:
        for token in vocab_tokens:
            try:
                idx = vocab[token]
                embedding_weights[idx] = vector_model.wv[token]
            except KeyError:
                logger('KeyError' + token, 0)
                pass

    embedding = embedding_layer(input_sequence_size=max_seq_length,
                                embedding_weights=embedding_weights,
                                vocab_size=vocab_size)

    return build_keras_lstm(embedding, output_size, dropout, learning_rate)


def build_keras_lstm(embedding, output_size, dropout, learning_rate):
    model = Sequential()
    model.add(embedding)
    model.add(SpatialDropout1D(dropout))

    if USE_CONV:
        model.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))
        model.add(MaxPooling1D(pool_size=2))

    model.add(Bidirectional(LSTM(WORD2VEC_PARAMS['size'], dropout=dropout, recurrent_dropout=dropout)))
    model.add(BatchNormalization())
    model.add(Dense(output_size, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer=Adam(lr=learning_rate))
    return model


def embedding_layer(vocab_size, input_sequence_size, embedding_weights):
    embedding_arguments = {
        'input_dim': vocab_size,
        'input_length': input_sequence_size,
        'mask_zero': False,  # index 0 reserved for padding
        'output_dim': WORD2VEC_PARAMS['size'],
        'trainable': TRAINABLE_EMBEDDING,
        'weights': [embedding_weights]
    }

    return Embedding(**embedding_arguments)


def logger(text, level=0):
    if level >= LOG_LEVEL:
        print(text)


def plot_history(history):
    plt_title = LOG_DIR
    plt.figure(figsize=(5, 5))
    plt.plot(history["loss"], 'g', label="training")
    plt.plot(history["val_loss"], 'r', label="validation")
    plt.yticks(np.arange(0, 1, 0.05))
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend(loc='upper right')
    plt.title(plt_title)
    plt.show()


def get_run_config():
    return {
        'STEM': STEM,
        'BATCH_SIZE': BATCH_SIZE,
        'DROPOUT': DROPOUT,
        'EPOCHS': EPOCHS,
        'LEARNING_RATE': LEARNING_RATE,
        'WORD2VEC_PARAMS': WORD2VEC_PARAMS,
        'TEST_SPLIT': TEST_SPLIT,
        'VALIDATION_SPLIT': VALIDATION_SPLIT,
        'USE_W2V_EMBED': USE_W2V_EMBED,
        'TRAINABLE_EMBEDDING': TRAINABLE_EMBEDDING,
        'USE_CONV': USE_CONV,
        'CROSS_VALIDATION': CROSS_VALIDATION
    }


def save_vocab_model_report(log_dir, vocab, model, report, run_config):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    with open(os.path.join(log_dir, 'vocab.pickle'), 'wb') as fp:
        pickle.dump(vocab, fp)

    with open(os.path.join(log_dir, 'model.pickle'), 'wb') as fp:
        pickle.dump(model, fp)

    with open(os.path.join(log_dir, 'classification_report.pickle'), 'wb') as fp:
        pickle.dump(report, fp)

    with open(os.path.join(log_dir, 'run_config.json'), 'w') as fp:
        json.dump(run_config, fp)


def get_quora_duplicate_tokens(limit=False):
    df = pd.DataFrame.from_csv(QUORA_DATA_DIR, sep='\t')
    duplicates = df[df['is_duplicate'] == 1]

    if limit:
        question_1 = duplicates['question1'][:limit]
        question_2 = duplicates['question2'][:limit]
    else:
        question_1 = duplicates['question1']
        question_2 = duplicates['question2']

    tokens_1 = [text_to_tokens(text) for text in question_1]
    tokens_2 = [text_to_tokens(text) for text in question_2]
    return tokens_1, tokens_2


def get_max_seq_length(model):
    return model.layers[0].get_output_at(0).get_shape().as_list()[1]


def get_q_classes():
    return Q_CLASSES


def tokens_to_sequences(vocab_idx_dict, texts_tokens):
    return [[vocab_idx_dict[token] for token in tokens] for tokens in texts_tokens]


def train_test_model(model, x_train, x_test, y_train, y_test, epochs, batch_size, log_dir):
    callbacks = [
        EarlyStopping(monitor='val_loss', min_delta=0, patience=5, verbose=0, mode='auto'),
        TensorBoard(log_dir=log_dir, histogram_freq=0, write_graph=True, write_images=True)
    ]
    # Train the model
    hist = model.fit(x_train, y_train, callbacks=callbacks, validation_split=VALIDATION_SPLIT, epochs=epochs, batch_size=batch_size, shuffle=True)
    # Test the model
    plot_history(pd.DataFrame(hist.history))
    y_hat = model.predict(x_test)
    report = classification_report(np.argmax(y_test, axis=1), np.argmax(y_hat, axis=1), target_names=get_q_classes(), output_dict=False)
    logger(report, 1)
    return model, report


def run(quora_tokens=False):
    # Original tokens to classify
    q_texts_tokens, q_labels = get_q_texts_tokens()
    # Additional tokens to use later
    if not quora_tokens:
        quora_tokens_q_1, quora_tokens_q_2 = get_quora_duplicate_tokens(limit=False)
        quora_tokens = quora_tokens_q_1 + quora_tokens_q_2
    # Embedding creation
    all_tokens = q_texts_tokens + quora_tokens
    vocab = build_vocab(all_tokens)
    sequences = tokens_to_sequences(vocab, q_texts_tokens)
    max_sequence_length = max([len(tokens) for tokens in all_tokens])
    model = build_model(vocab, max_sequence_length, len(get_q_classes()), USE_W2V_EMBED, DROPOUT, LEARNING_RATE)
    logger(model.summary())
    # build x and y
    x = pad_sequences(sequences, maxlen=get_max_seq_length(model), padding="pre", truncating="post")
    y = one_hot_encode_labels(q_labels)
    # Train test split
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=TEST_SPLIT, shuffle=False)
    model, report = train_test_model(model, x_train, x_test, y_train, y_test, EPOCHS, BATCH_SIZE, LOG_DIR)

    logger(report, 1)
    save_vocab_model_report(LOG_DIR, vocab, model, report, get_run_config())
    return vocab, model


if __name__ == "__main__":
    run()

# Cross validation code
# if CROSS_VALIDATION:
#     k_fold = StratifiedKFold(n_splits=10, shuffle=True)
#     report = {
#         'factoid': {'precision': [], 'recall': [], 'f1-score': [], 'support': []},
#         'list': {'precision': [], 'recall': [], 'f1-score': [], 'support': []},
#         'summary': {'precision': [], 'recall': [], 'f1-score': [], 'support': []},
#         'yesno': {'precision': [], 'recall': [], 'f1-score': [], 'support': []}
#     }
#     target_names = get_q_classes()
#     metrics = ['precision', 'recall', 'f1-score', 'support']
#
#     for train, test in k_fold.split(x, y):
#         y_train = y_encoded[train]
#         y_test = y_encoded[test]
#         hist = model.fit(x[train], y_train, callbacks=callbacks, validation_split=VALIDATION_SPLIT,
#                                               epochs=EPOCHS, batch_size=BATCH_SIZE)
#         plot_history(pd.DataFrame(hist.history))
#         y_hat = model.predict(x[test])
#         class_report = classification_report(np.argmax(y_test, axis=1), np.argmax(y_hat, axis=1),
#                                                       target_names=target_names, output_dict=True)
#
#         for target in target_names:
#             for metric in metrics:
#                 report[target][metric].append(class_report[target][metric])
#
#     for target in target_names:
#         for metric in metrics:
#             cv_scores = report[target][metric]
#             report[target][metric] = "%.2f%% (+/- %.2f%%)" % (np.mean(cv_scores), np.std(cv_scores))
# else: