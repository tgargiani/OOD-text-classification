import os, json
from utils import Split_BERT, DS_INCOMPLETE_PATH
from transformers import TFBertForSequenceClassification, BertTokenizer, BertConfig
import tensorflow as tf
import numpy as np
from keras.preprocessing.sequence import pad_sequences


def tokenize_sentences(sentences, tokenizer, max_seq_len=128):
    tokenized_sentences = []

    for sentence in sentences:
        tokenized_sentence = tokenizer.encode(
            sentence,  # Sentence to encode.
            add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
            max_length=max_seq_len,  # Truncate all sentences.
        )

        tokenized_sentences.append(tokenized_sentence)

    return tokenized_sentences


def create_attention_masks(tokenized_and_padded_sentences):
    attention_masks = []

    for sentence in tokenized_and_padded_sentences:
        att_mask = [int(token_id > 0) for token_id in sentence]
        attention_masks.append(att_mask)

    return np.asarray(attention_masks)


def create_dataset(ids, masks, labels):
    def gen():
        for i in range(len(ids)):
            yield (
                {
                    "input_ids": ids[i],
                    "attention_mask": masks[i]
                },
                labels[i],
                # tf.reshape(tf.constant(labels[i]), [1, num_labels]),
                # tf.reshape(tf.constant(labels[i]), [1, 151]),
            )

    return tf.data.Dataset.from_generator(
        gen,
        ({"input_ids": tf.int32, "attention_mask": tf.int32}, tf.int64),
        (
            {
                "input_ids": tf.TensorShape([None]),
                "attention_mask": tf.TensorShape([None])
            },
            tf.TensorShape([None]),
        ),
    )


def evaluate(dataset, split):
    # bert_model_name = 'bert-base-uncased'
    #
    # tokenizer = BertTokenizer.from_pretrained(bert_model_name, do_lower_case=True)
    # MAX_LEN = 128
    #
    # X_train, y_train = split.get_X_y(dataset['train'] + dataset['oos_train'])
    #
    # train_ids = tokenize_sentences(X_train, tokenizer, MAX_LEN)
    # train_ids = pad_sequences(train_ids, maxlen=MAX_LEN, dtype="long", value=0, truncating="post", padding="post")
    # train_masks = create_attention_masks(train_ids)
    #
    # train_dataset = create_dataset(train_ids, train_masks, y_train)
    #
    # model = TFBertForSequenceClassification.from_pretrained(
    #     bert_model_name,
    #     config=BertConfig.from_pretrained(bert_model_name, num_labels=151)
    # )
    #
    # # Prepare training: Compile tf.keras model with optimizer, loss and learning rate schedule
    # optimizer = tf.keras.optimizers.Adam(learning_rate=3e-5, epsilon=1e-08, clipnorm=1.0)
    # loss = tf.keras.losses.SparseCategoricalCrossentropy()#from_logits=True)
    # metric = tf.keras.metrics.SparseCategoricalCrossentropy('accuracy')
    # model.compile(optimizer=optimizer, loss=loss, metrics=[metric])
    #
    # # Train and evaluate using tf.keras.Model.fit()
    # history = model.fit(train_dataset, epochs=1, steps_per_epoch=115, validation_steps=7)





    config = BertConfig.from_pretrained('bert-base-uncased')
    config.num_labels = 151
    print(config)

    model = TFBertForSequenceClassification(config)

    X_train, y_train = split.get_X_y(dataset['train'] + dataset['oos_train'])
    X_test, y_test = split.get_X_y(dataset['test'] + dataset['oos_test'])

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    train_encodings = tokenizer(X_train, truncation=True, padding=True)
    test_encodings = tokenizer(X_test, truncation=True, padding=True)

    train_dataset = tf.data.Dataset.from_tensor_slices((
        dict(train_encodings),
        y_train
    ))
    test_dataset = tf.data.Dataset.from_tensor_slices((
        dict(test_encodings),
        y_test
    ))

    optimizer = tf.keras.optimizers.Adam(learning_rate=3e-5)
    loss = tf.keras.losses.SparseCategoricalCrossentropy()  # config)#from_logits=True)
    model.compile(optimizer=optimizer, loss=loss)
    model.fit(train_dataset.batch(32), epochs=2, steps_per_epoch=115)

    model.predict(test_dataset)


if __name__ == '__main__':
    RANDOM_SELECTION = False  # am I testing using the random selection of IN intents?

    for dataset_size in ['data_full']:  # , 'data_small', 'data_imbalanced', 'data_oos_plus']:
        print(f'Testing on: {dataset_size}')

        path_intents = os.path.join(DS_INCOMPLETE_PATH, dataset_size + '.json')

        with open(path_intents) as f:  # open intent dataset
            int_ds = json.load(f)

        split = Split_BERT()
        evaluate(int_ds, split)

        # if not RANDOM_SELECTION:
        #     results_dct = evaluate(int_ds, DIM)
        #
        #     print_results(dataset_size, results_dct)
