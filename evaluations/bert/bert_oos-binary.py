import os, json, copy
from utils import DS_INCOMPLETE_PATH, Split_BERT, tokenize_BERT, get_filtered_lst, print_results, get_intents_selection, \
    save_results
from testing import Testing
from transformers import TFBertForSequenceClassification, BertTokenizer
import tensorflow as tf
from numpy import mean


def evaluate(binary_dataset, model_int, X_int_test, y_int_test, split):
    # Split and tokenize dataset
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    X_bin_train, y_bin_train = split.get_X_y(binary_dataset['train'], limit_num_sents=False,
                                             set_type='train')
    X_bin_val, y_bin_val = split.get_X_y(binary_dataset['val'], limit_num_sents=False, set_type='val')

    train_bin_ids, train_bin_attention_masks, train_bin_labels = tokenize_BERT(X_bin_train, y_bin_train, tokenizer)
    val_bin_ids, val_bin_attention_masks, val_bin_labels = tokenize_BERT(X_bin_val, y_bin_val, tokenizer)

    # Train model
    model_bin = TFBertForSequenceClassification.from_pretrained('bert-base-uncased',
                                                                num_labels=2)  # we have to adjust the number of labels
    print('\nBert Model', model_bin.summary())

    log_dir = 'tensorboard_data/tb_bert_bin'
    model_save_path = './models/bert_model_bin.h5'

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(filepath=model_save_path, save_weights_only=True, monitor='val_loss',
                                           mode='min',
                                           save_best_only=True), tf.keras.callbacks.TensorBoard(log_dir=log_dir)]

    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    metric = tf.keras.metrics.SparseCategoricalAccuracy('accuracy')
    optimizer = tf.keras.optimizers.Adam(learning_rate=2e-5)

    model_bin.compile(loss=loss, optimizer=optimizer, metrics=[metric])

    history = model_bin.fit([train_bin_ids, train_bin_attention_masks],
                            train_bin_labels,
                            batch_size=128,
                            epochs=4,
                            validation_data=([val_bin_ids, val_bin_attention_masks], val_bin_labels),
                            callbacks=callbacks)

    # Test
    testing = Testing(model_int, X_int_test, y_int_test, 'bert', split.intents_dct['oos'], bin_model=model_bin)
    results_dct = testing.test_binary()

    return results_dct


def train_intent_model(int_ds, random_selection: bool, limit_num_sents: bool, num_samples=None):
    if random_selection:
        selection = get_intents_selection(int_ds['train'],
                                          num_samples)  # selected intent labels: (num_samples, ) np.ndarray

        filt_train = get_filtered_lst(int_ds['train'],
                                      selection)  # almost the same as int_ds['train'] but filtered according to selection
        filt_test = get_filtered_lst(int_ds['test'], selection)

        mod_int_ds = copy.deepcopy(int_ds)  # deepcopy in order to not modify the original dict
        mod_int_ds['train'] = filt_train
        mod_int_ds['test'] = filt_test

        dataset = mod_int_ds
    else:
        dataset = int_ds

    # Split and tokenize dataset
    split = Split_BERT()
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    X_int_train, y_int_train = split.get_X_y(dataset['train'], limit_num_sents=limit_num_sents, set_type='train')
    X_int_val, y_int_val = split.get_X_y(dataset['val'], limit_num_sents=limit_num_sents, set_type='val')
    X_int_test, y_int_test = split.get_X_y(dataset['test'] + dataset['oos_test'], limit_num_sents=limit_num_sents,
                                           set_type='test')

    train_int_ids, train_int_attention_masks, train_int_labels = tokenize_BERT(X_int_train, y_int_train, tokenizer)
    val_int_ids, val_int_attention_masks, val_int_labels = tokenize_BERT(X_int_val, y_int_val, tokenizer)

    num_labels = len(split.intents_dct.keys()) - 1  # minus 1 because 'oos' label isn't used in training

    # Train model
    model_int = TFBertForSequenceClassification.from_pretrained('bert-base-uncased',
                                                                num_labels=num_labels)  # we have to adjust the number of labels
    print('\nBert Model', model_int.summary())

    log_dir = 'tensorboard_data/tb_bert'
    model_save_path = './models/bert_model.h5'

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(filepath=model_save_path, save_weights_only=True, monitor='val_loss',
                                           mode='min',
                                           save_best_only=True), tf.keras.callbacks.TensorBoard(log_dir=log_dir)]

    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    metric = tf.keras.metrics.SparseCategoricalAccuracy('accuracy')
    optimizer = tf.keras.optimizers.Adam(learning_rate=2e-5)

    model_int.compile(loss=loss, optimizer=optimizer, metrics=[metric])

    history = model_int.fit([train_int_ids, train_int_attention_masks],
                            train_int_labels,
                            batch_size=128,
                            epochs=4,
                            validation_data=([val_int_ids, val_int_attention_masks], val_int_labels),
                            callbacks=callbacks)

    return model_int, X_int_test, y_int_test, split


if __name__ == '__main__':
    RANDOM_SELECTION = True  # am I testing using the random selection of IN intents?
    repetitions = 30  # number of evaluations when using random selection
    LIMIT_NUM_SENTS = True  # am I limiting the number of sentences of each intent?

    # Intent classifier
    path_intents = os.path.join(DS_INCOMPLETE_PATH, 'data_full.json')  # always use data_full dataset

    with open(path_intents) as f:
        int_ds = json.load(f)

    for dataset_size in ['binary_undersample', 'binary_wiki_aug']:
        print(f'Testing on: {dataset_size}\n')

        # Binary classifier
        path_bin = os.path.join(DS_INCOMPLETE_PATH, dataset_size + '.json')

        with open(path_bin) as f:  # open binary intent dataset
            bin_ds = json.load(f)

        if not RANDOM_SELECTION:
            model_int, X_int_test, y_int_test, split = train_intent_model(int_ds, RANDOM_SELECTION, LIMIT_NUM_SENTS)

            results_dct = evaluate(bin_ds, model_int, X_int_test, y_int_test, split)

            print_results(dataset_size, results_dct)
        else:
            for num_samples in [3, 6, 9, 12]:  # choose only a certain number of samples
                print(f'{repetitions} times random selection {num_samples} intents')

                accuracy_lst, recall_lst = [], []
                far_lst, frr_lst = [], []

                for i in range(repetitions):
                    model_int, X_int_test, y_int_test, split = train_intent_model(int_ds, RANDOM_SELECTION,
                                                                                  LIMIT_NUM_SENTS, num_samples)

                    temp_res = evaluate(bin_ds, model_int, X_int_test, y_int_test, split)  # temporary results

                    accuracy_lst.append(temp_res['accuracy'])
                    recall_lst.append(temp_res['recall'])
                    far_lst.append(temp_res['far'])
                    frr_lst.append(temp_res['frr'])

                results_dct = {}  # computed as mean of all temporary results
                results_dct['accuracy'] = float(mean(accuracy_lst))
                results_dct['recall'] = float(mean(recall_lst))
                results_dct['far'] = float(mean(far_lst))
                results_dct['frr'] = float(mean(frr_lst))

                # save_results('bert', 'oos-binary', dataset_size, num_samples, repetitions,
                #              {'accuracy_lst': accuracy_lst, 'recall_lst': recall_lst, 'far_lst': far_lst,
                #               'frr_lst': frr_lst}, results_dct)

                print_results(dataset_size, results_dct)

        print('------------------------------------\n')
