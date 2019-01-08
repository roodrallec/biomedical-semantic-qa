import pickle
import part_1
import numpy as np
import os
from keras.preprocessing.sequence import pad_sequences

FROM_CACHE = True
MODEL_DIR = "./Graph/2019-01-07 11:34:17.175065/model.pickle"
VOCAB_DIR = "./Graph/2019-01-07 11:34:17.175065/vocab.pickle"
EPOCHS = 100
BATCH_SIZE = 128
SEQUENCE_LIMIT = 50000
LEARNING_RATE = 0.001
DROPOUT = 0.3
ALLOWED_CLASS = ['factoid']


def load_pickle(file_path):
    with open(file_path, 'rb') as f:
        return pickle.load(f)


def get_max_seq_length(model):
    return model.layers[0].get_output_at(0).get_shape().as_list()[1]


def build_new_tokens(question_1, question_1_predictions, question_2, question_2_predictions, q_classes):
    print("Building new tokens")
    new_tokens = []
    new_labels = []

    for idx, prediction_1 in enumerate(question_1_predictions):
        prediction_2 = question_2_predictions[idx]
        _class = q_classes[prediction_2]

        if prediction_1 == prediction_2 and _class in ALLOWED_CLASS:
            new_tokens.append(question_1[idx])
            new_labels.append(_class)
            new_tokens.append(question_2[idx])
            new_labels.append(_class)

    return new_tokens, new_labels


def run():
    if FROM_CACHE and MODEL_DIR and VOCAB_DIR:
        model = load_pickle(MODEL_DIR)
        vocab = load_pickle(VOCAB_DIR)
    else:
        vocab, model = part_1.run()

    print("LOADING QUORA DUPLICATES")
    question_1, question_2 = part_1.get_quora_duplicate_tokens(SEQUENCE_LIMIT)
    max_seq_length = get_max_seq_length(model)
    question_1_x = pad_sequences(part_1.tokens_to_sequences(vocab, question_1),
                                 maxlen=max_seq_length, padding="pre", truncating="post")
    question_2_x = pad_sequences(part_1.tokens_to_sequences(vocab, question_2),
                                 maxlen=max_seq_length, padding="pre", truncating="post")

    print("MAKING PREDICTIONS")
    question_1_predictions = np.argmax(model.predict(question_1_x), axis=1)
    question_2_predictions = np.argmax(model.predict(question_2_x), axis=1)

    print("BUILDING NEW MODEL")
    old_tokens, old_labels = part_1.get_q_texts_tokens()
    q_classes = part_1.get_q_classes(old_labels)

    new_tokens, new_labels = build_new_tokens(
        question_1, question_1_predictions,
        question_2, question_2_predictions,
        q_classes)


    new_tokens = old_tokens + new_tokens
    new_labels = list(old_labels) + list(new_labels)
    new_model = part_1.build_model(
        new_tokens,
        vocab,
        max_seq_length,
        len(q_classes),
        True,
        DROPOUT,
        LEARNING_RATE
    )
    new_sequences = part_1.tokens_to_sequences(vocab, new_tokens)

    print("TRAINING NEW MODEL")

    new_trained_model, new_report = part_1.train_test_model(
        new_model,
        new_sequences,
        new_labels,
        EPOCHS,
        BATCH_SIZE,
        test_idx=int(0.1*len(old_tokens))
    )

    part_1.save_vocab_model_report('.', vocab, new_trained_model, new_report)


if __name__ == "__main__":
    run()