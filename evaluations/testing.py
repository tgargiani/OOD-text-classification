import numpy as np
import tensorflow as tf
from transformers import BertTokenizer


# TP = predicted as OOD and true label is OOD
# TN = predicted as IN and true label is IN
# FP = predicted as OOD and true label is IN
# FN = predicted as IN and true label is OOD

# FAR = Number of accepted OOD sentences / Number of OOD sentences
# FAR = FN / (TP + FN)

# FRR = Number of rejected ID sentences / Number of ID sentences
# FRR = FP / (FP + TN)


class Testing:
    """Used to test the results of classification."""

    def __init__(self, model, X_test, y_test, model_type: str, oos_label, bin_model=None, bin_oos_label=None):
        self.model = model
        self.X_test = X_test  # list or dict with 'test_ids' and 'test_attention_masks' as keys (in case of BERT train/threshold)
        self.y_test = y_test  # list
        self.oos_label = oos_label  # number or string
        self.model_type = model_type
        self.bin_model = bin_model
        self.bin_oos_label = bin_oos_label  # BERT needs a different 'oos' label in binary approach

    def test_train(self):
        accuracy_correct, accuracy_out_of = 0, 0
        recall_correct, recall_out_of = 0, 0

        tp, tn, fp, fn = 0, 0, 0, 0

        if self.model_type in ['fasttext', 'svm', 'mlp']:
            predictions = self.model.predict(self.X_test)
        elif self.model_type == 'bert':
            tf_output = self.model.predict([self.X_test['test_ids'], self.X_test['test_attention_masks']])
            tf_output = tf_output[0]
            predictions = tf.nn.softmax(tf_output, axis=1).numpy()

        # unify different output formats of various model.predict() functions
        if self.model_type == 'fasttext':
            pred_labels = [label[0] for label in predictions[0]]
        elif self.model_type in ['svm', 'mlp']:
            pred_labels = predictions
        elif self.model_type == 'bert':
            pred_labels = np.argmax(predictions, axis=1)

        # Rasa can't predict all intents at once
        if self.model_type == 'rasa':
            pred_labels = []

            for sent in self.X_test:
                pred = self.model.parse(sent)
                pred_labels.append(pred['intent']['name'])

        for pred_label, true_label in zip(pred_labels, self.y_test):

            # the following set of conditions is the same for all testing methods
            if true_label != self.oos_label:
                if pred_label == true_label:
                    accuracy_correct += 1

                if pred_label != self.oos_label:
                    tn += 1
                else:
                    fp += 1

                accuracy_out_of += 1
            else:
                if pred_label == true_label:
                    recall_correct += 1
                    tp += 1
                else:
                    fn += 1

                recall_out_of += 1

        accuracy = accuracy_correct / accuracy_out_of * 100
        recall = recall_correct / recall_out_of * 100

        far = fn / (tp + fn) * 100  # false acceptance rate
        frr = fp / (fp + tn) * 100  # false recognition rate

        return {'accuracy': accuracy, 'recall': recall, 'far': far, 'frr': frr}

    def test_threshold(self, threshold: float):
        accuracy_correct, accuracy_out_of = 0, 0
        recall_correct, recall_out_of = 0, 0

        tp, tn, fp, fn = 0, 0, 0, 0

        # unify different output formats of various model.predict() functions
        if self.model_type == 'fasttext':
            predictions = self.model.predict(self.X_test)

            pred_labels = [label[0] for label in predictions[0]]
            pred_similarities = [similarity[0] for similarity in predictions[1]]
        elif self.model_type in ['svm', 'mlp']:
            prediction_probabilities = self.model.predict_proba(self.X_test)  # intent prediction probabilities

            pred_labels = np.argmax(prediction_probabilities, axis=1)  # intent predictions
            pred_similarities = np.take_along_axis(prediction_probabilities, np.expand_dims(pred_labels, axis=-1),
                                                   axis=-1).squeeze(axis=-1)
        elif self.model_type == 'bert':
            tf_output = self.model.predict([self.X_test['test_ids'], self.X_test['test_attention_masks']])
            tf_output = tf_output[0]
            prediction_probabilities = tf.nn.softmax(tf_output, axis=1).numpy()

            pred_labels = np.argmax(prediction_probabilities, axis=1)
            pred_similarities = np.take_along_axis(prediction_probabilities, np.expand_dims(pred_labels, axis=-1),
                                                   axis=-1).squeeze(axis=-1)

        # Rasa can't predict all intents at once
        if self.model_type == 'rasa':
            pred_labels = []
            pred_similarities = []

            for sent in self.X_test:
                pred = self.model.parse(sent)
                pred_label = pred['intent']['name']
                pred_similarity = pred['intent']['confidence']

                pred_labels.append(pred_label)
                pred_similarities.append(pred_similarity)

        for pred_label, pred_similarity, true_label in zip(pred_labels, pred_similarities, self.y_test):

            if pred_similarity < threshold:
                pred_label = self.oos_label

            # the following set of conditions is the same for all testing methods
            if true_label != self.oos_label:
                if pred_label == true_label:
                    accuracy_correct += 1

                if pred_label != self.oos_label:
                    tn += 1
                else:
                    fp += 1

                accuracy_out_of += 1
            else:
                if pred_label == true_label:
                    recall_correct += 1
                    tp += 1
                else:
                    fn += 1

                recall_out_of += 1

        accuracy = accuracy_correct / accuracy_out_of * 100
        recall = recall_correct / recall_out_of * 100

        far = fn / (tp + fn) * 100  # false acceptance rate
        frr = fp / (fp + tn) * 100  # false recognition rate

        return {'accuracy': accuracy, 'recall': recall, 'far': far, 'frr': frr}

    def test_binary(self):
        accuracy_correct, accuracy_out_of = 0, 0
        recall_correct, recall_out_of = 0, 0

        tp, tn, fp, fn = 0, 0, 0, 0

        if self.model_type == 'bert':
            # create tokenizer in order to predict single sentences
            tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

        for sent, true_label in zip(self.X_test, self.y_test):
            if self.model_type == 'bert':
                predict_input = tokenizer.encode(sent,
                                                 truncation=True,
                                                 padding=True,
                                                 return_tensors="tf")

            # 1st step - binary classification
            if self.model_type == 'fasttext':
                bin_pred = self.bin_model.predict(sent)

                bin_pred_label = bin_pred[0][0]
            elif self.model_type in ['svm', 'mlp']:
                bin_pred = self.bin_model.predict(sent)

                bin_pred_label = bin_pred[0]
            elif self.model_type == 'bert':
                bin_tf_output = self.bin_model.predict(predict_input)[0]
                bin_pred_probs = tf.nn.softmax(bin_tf_output, axis=1).numpy()[0]

                bin_pred_label = np.argmax(bin_pred_probs)
            elif self.model_type == 'rasa':
                bin_pred = self.bin_model.parse(sent)

                bin_pred_label = bin_pred['intent']['name']

            if bin_pred_label != self.oos_label or bin_pred_label != self.bin_oos_label:
                # 2nd step - intent classification
                if self.model_type == 'fasttext':
                    int_pred = self.model.predict(sent)

                    int_pred_label = int_pred[0][0]
                elif self.model_type in ['svm', 'mlp']:
                    int_pred = self.model.predict(sent)

                    int_pred_label = int_pred[0]
                elif self.model_type == 'bert':
                    int_tf_output = self.model.predict(predict_input)[0]
                    int_pred_probs = tf.nn.softmax(int_tf_output, axis=1).numpy()[0]

                    int_pred_label = np.argmax(int_pred_probs)
                elif self.model_type == 'rasa':
                    int_pred = self.model.parse(sent)

                    int_pred_label = int_pred['intent']['name']

                pred_label = int_pred_label
            else:
                pred_label = self.oos_label

            # the following set of conditions is the same for all testing methods
            if true_label != self.oos_label:
                if pred_label == true_label:
                    accuracy_correct += 1

                if pred_label != self.oos_label:
                    tn += 1
                else:
                    fp += 1

                accuracy_out_of += 1
            else:
                if pred_label == true_label:
                    recall_correct += 1
                    tp += 1
                else:
                    fn += 1

                recall_out_of += 1

        accuracy = accuracy_correct / accuracy_out_of * 100
        recall = recall_correct / recall_out_of * 100

        far = fn / (tp + fn) * 100  # false acceptance rate
        frr = fp / (fp + tn) * 100  # false recognition rate

        return {'accuracy': accuracy, 'recall': recall, 'far': far, 'frr': frr}
