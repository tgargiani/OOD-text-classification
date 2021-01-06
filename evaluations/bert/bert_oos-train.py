import os, json, copy
from utils import DS_INCOMPLETE_PATH, Split_BERT, tokenize_BERT, get_filtered_lst, print_results, get_intents_selection, \
    save_results
from testing import Testing
from transformers import TFBertForSequenceClassification, BertTokenizer
import tensorflow as tf
from numpy import mean


def evaluate(dataset, limit_num_sents: bool):
    # Split and tokenize dataset
    split = Split_BERT()
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    X_train, y_train = split.get_X_y(dataset['train'] + dataset['oos_train'], limit_num_sents=limit_num_sents,
                                     set_type='train')
    X_val, y_val = split.get_X_y(dataset['val'] + dataset['oos_val'], limit_num_sents=limit_num_sents, set_type='val')
    X_test, y_test = split.get_X_y(dataset['test'] + dataset['oos_test'], limit_num_sents=limit_num_sents,
                                   set_type='test')

    train_ids, train_attention_masks, train_labels = tokenize_BERT(X_train, y_train, tokenizer)
    val_ids, val_attention_masks, val_labels = tokenize_BERT(X_val, y_val, tokenizer)
    test_ids, test_attention_masks, test_labels = tokenize_BERT(X_test, y_test, tokenizer)

    num_labels = len(split.intents_dct.keys())

    # Train model
    model = TFBertForSequenceClassification.from_pretrained('bert-base-uncased',
                                                            num_labels=num_labels)  # we have to adjust the number of labels
    print('\nBert Model', model.summary())

    log_dir = 'tensorboard_data/tb_bert'
    model_save_path = './models/bert_model.h5'

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(filepath=model_save_path, save_weights_only=True, monitor='val_loss',
                                           mode='min',
                                           save_best_only=True), tf.keras.callbacks.TensorBoard(log_dir=log_dir)]

    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    metric = tf.keras.metrics.SparseCategoricalAccuracy('accuracy')
    optimizer = tf.keras.optimizers.Adam(learning_rate=2e-5)

    model.compile(loss=loss, optimizer=optimizer, metrics=[metric])

    history = model.fit([train_ids, train_attention_masks],
                        train_labels,
                        batch_size=128,
                        epochs=4,
                        validation_data=([val_ids, val_attention_masks], val_labels),
                        callbacks=callbacks)

    # Test
    testing = Testing(model, {'test_ids': test_ids, 'test_attention_masks': test_attention_masks}, test_labels,
                      'bert', split.intents_dct['oos'])
    results_dct = testing.test_train()

    return results_dct


if __name__ == '__main__':
    RANDOM_SELECTION = True  # am I testing using the random selection of IN intents?
    repetitions = 30  # number of evaluations when using random selection
    LIMIT_NUM_SENTS = True  # am I limiting the number of sentences of each intent?

    for dataset_size in ['data_full']:  # , 'data_small', 'data_imbalanced', 'data_oos_plus']:
        print(f'Testing on: {dataset_size}')

        path_intents = os.path.join(DS_INCOMPLETE_PATH, dataset_size + '.json')

        with open(path_intents) as f:  # open intent dataset
            int_ds = json.load(f)

        if not RANDOM_SELECTION:
            results_dct = evaluate(int_ds, LIMIT_NUM_SENTS)

            print_results(dataset_size, results_dct)
        else:
            for num_samples in [3, 6, 9, 12]:  # choose only a certain number of samples
                print(f'{repetitions} times random selection {num_samples} intents')

                accuracy_lst, recall_lst = [], []
                far_lst, frr_lst = [], []

                for i in range(repetitions):
                    selection = get_intents_selection(int_ds['train'],
                                                      num_samples)  # selected intent labels: (num_samples, ) np.ndarray

                    filt_train = get_filtered_lst(int_ds['train'],
                                                  selection)  # almost the same as int_ds['train'] but filtered according to selection
                    filt_val = get_filtered_lst(int_ds['val'], selection)
                    filt_test = get_filtered_lst(int_ds['test'], selection)

                    mod_int_ds = copy.deepcopy(int_ds)  # deepcopy in order to not modify the original dict
                    mod_int_ds['train'] = filt_train
                    mod_int_ds['val'] = filt_val
                    mod_int_ds['test'] = filt_test

                    temp_res = evaluate(mod_int_ds, LIMIT_NUM_SENTS)  # temporary results

                    accuracy_lst.append(temp_res['accuracy'])
                    recall_lst.append(temp_res['recall'])
                    far_lst.append(temp_res['far'])
                    frr_lst.append(temp_res['frr'])

                results_dct = {}  # computed as mean of all temporary results
                results_dct['accuracy'] = float(mean(accuracy_lst))
                results_dct['recall'] = float(mean(recall_lst))
                results_dct['far'] = float(mean(far_lst))
                results_dct['frr'] = float(mean(frr_lst))

                # save_results('bert', 'oos-train', dataset_size, num_samples, repetitions,
                #              {'accuracy_lst': accuracy_lst, 'recall_lst': recall_lst, 'far_lst': far_lst,
                #               'frr_lst': frr_lst}, results_dct)

                print_results(dataset_size, results_dct)

        print('------------------------------------\n')
