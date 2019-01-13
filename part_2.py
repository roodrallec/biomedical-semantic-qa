import pickle
import part_1
import numpy as np
import os
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split

# RUN CONFIG
RUN_CONFIG = {
    "FROM_CACHE": False,
    "MODEL_DIR": "./Graphs_datetime/2019-01-11 17:35:16.028132/model.pickle",
    "VOCAB_DIR": "./Graphs_datetime/2019-01-11 17:35:16.028132/vocab.pickle",
    "EPOCHS": 50,
    "BATCH_SIZE": 512,
    "EXTRA_TOKENS": 50000,
    "LEARNING_RATE": 0.001,
    "DROPOUT": 0.5,
    "ALLOWED_CLASS": 'factoid'
}


def load_pickle(file_path):
    with open(file_path, 'rb') as f:
        return pickle.load(f)


def save_pickle(obj, file_path):
    with open(file_path, 'wb') as fp:
        pickle.dump(obj, fp)


def predict_labels(vocab, model, q_tokens_list, max_seq_length, class_label_map):
    sequence = pad_sequences([[vocab[token] for token in q_tokens] for q_tokens in q_tokens_list], maxlen=max_seq_length, padding="pre", truncating="post")
    classes = np.argmax(model.predict(sequence), axis=1)
    return [class_label_map[_class] for _class in classes]


def build_new_tokens(q_1_tokens_list, q_2_tokens_list, q_1_pred_labels, q_2_pred_labels, allowed_class, token_limit):
    new_tokens = []
    new_labels = []
    limit = 0

    for q_1_tokens, q_2_tokens, q_1_pred_label, q_2_pred_label in zip(q_1_tokens_list, q_2_tokens_list, q_1_pred_labels, q_2_pred_labels):

        if limit >= token_limit:
            break

        if allowed_class and q_1_pred_label not in allowed_class:
            continue

        if q_1_pred_label == q_2_pred_label:
            new_tokens.append(q_1_tokens)
            new_labels.append(q_1_pred_label)
            new_tokens.append(q_2_tokens)
            new_labels.append(q_2_pred_label)
            limit += 2

    return new_tokens, new_labels


def token_combine_split(vocab, old_tokens, old_labels, new_tokens, new_labels, max_seq_len):
    train_tokens, test_tokens, train_labels, test_labels = train_test_split(old_tokens, old_labels, test_size=part_1.get_test_split(), shuffle=False)
    train_tokens = train_tokens + list(new_tokens)
    train_labels = train_labels + list(new_labels)
    train_sequences = pad_sequences(part_1.tokens_to_sequences(vocab, train_tokens), maxlen=max_seq_len, padding="pre", truncating="post")
    test_sequences = pad_sequences(part_1.tokens_to_sequences(vocab, test_tokens), maxlen=max_seq_len, padding="pre", truncating="post")

    return train_sequences, test_sequences, part_1.one_hot_encode_labels(train_labels), part_1.one_hot_encode_labels(test_labels)


def run():
    tokens_cache_name = 'quora_tokens.pickle'
    if RUN_CONFIG['FROM_CACHE'] and os.path.exists(tokens_cache_name):
        print("LOADING QUORA DUPLICATES AS TOKENS FROM CACHE")
        tokens_question_1, tokens_question_2 = load_pickle(tokens_cache_name)
    else:
        print("PARSING QUORA DUPLICATES AS TOKENS")
        tokens_question_1, tokens_question_2 = part_1.get_quora_duplicate_tokens()
        save_pickle([tokens_question_1, tokens_question_2], tokens_cache_name)

    if RUN_CONFIG['FROM_CACHE'] and RUN_CONFIG['MODEL_DIR'] and RUN_CONFIG['VOCAB_DIR']:
        print("LOADING ORIGINAL MODEL FROM CACHE")
        original_model = load_pickle(RUN_CONFIG['MODEL_DIR'])
        vocab = load_pickle(RUN_CONFIG['VOCAB_DIR'])
    else:
        print("BUILDING ORIGINAL MODEL")
        vocab, original_model = part_1.run(tokens_question_1 + tokens_question_2)

    print("LOADING ORIGINAL TOKENS")
    max_seq_length = part_1.get_max_seq_length(original_model)
    original_classes = part_1.get_q_classes()
    original_tokens, original_labels = part_1.get_q_texts_tokens()

    predict_cache_name = 'predictions.pickle'
    if RUN_CONFIG['FROM_CACHE'] and os.path.exists(predict_cache_name):
        print("LOADING QUORA DUPLICATE PREDICTIONS")
        q_1_predictions, q_2_predictions = load_pickle('predictions.pickle')
    else:
        print("PREDICTING QUORA DUPLICATES")
        q_1_predictions = predict_labels(vocab, original_model, tokens_question_1, max_seq_length, original_classes)
        q_2_predictions = predict_labels(vocab, original_model, tokens_question_2, max_seq_length, original_classes)
        save_pickle([q_1_predictions, q_2_predictions], predict_cache_name)

    for _class in ['factoid', 'summary']:  #
        RUN_CONFIG['ALLOWED_CLASS'] = _class

        for limit in [22000]:  #1000, 6000, 11000,
            RUN_CONFIG['LEARNING_RATE'] = limit
            print("BUILDING NEW TOKENS")
            new_tokens, new_labels = build_new_tokens(tokens_question_1, tokens_question_2, q_1_predictions, q_2_predictions, allowed_class=_class, token_limit=limit)
            x_train, x_test, y_train, y_test = token_combine_split(vocab, original_tokens, original_labels, new_tokens, new_labels, max_seq_length)
            print("TRAINING NEW MODEL")

            for rate in [0.0001, 0.001, 0.01]:
                RUN_CONFIG['LEARNING_RATE'] = rate

                for dropout in [0.3, 0.5, 0.7]:
                    RUN_CONFIG['DROPOUT'] = dropout

                    for batch_size in [1024]: #125, 512,
                        RUN_CONFIG['BATCH_SIZE'] = batch_size
                        log_dir = "./Graph_3/" + _class + "_" + str(limit) + "_" + str(rate) + "_" + str(dropout) + "_" + str(batch_size)
                        new_model = part_1.build_model(vocab, max_seq_length, len(original_classes), embed_w2v=True, dropout=dropout, learning_rate=rate)
                        new_trained_model, new_report = part_1.train_test_model(new_model, x_train, x_test, y_train, y_test,
                                                                                epochs=RUN_CONFIG['EPOCHS'],
                                                                                batch_size=RUN_CONFIG['BATCH_SIZE'],
                                                                                log_dir=log_dir)

                        part_1.save_vocab_model_report(log_dir, vocab, new_trained_model, new_report, RUN_CONFIG)


if __name__ == "__main__":
    run()
