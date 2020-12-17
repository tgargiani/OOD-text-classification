import os, json
from utils import Split_BERT, DS_INCOMPLETE_PATH
from transformers import TFBertForSequenceClassification, BertTokenizer, BertConfig
import tensorflow as tf
import numpy as np
from keras.preprocessing.sequence import pad_sequences


def evaluate(dataset):
    # Split dataset
    split = Split_BERT()

    X_train, y_train = split.get_X_y(dataset['train'] + dataset['oos_train'])
    X_test, y_test = split.get_X_y(dataset['test'] + dataset['oos_test'])

    print(len(split.intents_dct))

if __name__ == '__main__':
    RANDOM_SELECTION = False  # am I testing using the random selection of IN intents?
    repetitions = 30  # number of evaluations when using random selection

    for dataset_size in ['data_full']:  # , 'data_small', 'data_imbalanced', 'data_oos_plus']:
        print(f'Testing on: {dataset_size}')

        path_intents = os.path.join(DS_INCOMPLETE_PATH, dataset_size + '.json')

        with open(path_intents) as f:  # open intent dataset
            int_ds = json.load(f)

        if not RANDOM_SELECTION:
            results_dct = evaluate(int_ds)
        #
        #     print_results(dataset_size, results_dct)
