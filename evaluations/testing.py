import numpy as np


class Testing:
    """Used to test the results of classification."""

    def __init__(self, model, X_test, y_test, model_type: str, oos_label, bin_model=None):
        self.model = model
        self.X_test = X_test
        self.y_test = y_test
        self.oos_label = oos_label
        self.model_type = model_type
        self.bin_model = bin_model

    def test_train(self):
        accuracy_correct = 0
        accuracy_out_of = 0

        recall_correct = 0
        recall_out_of = 0

        predictions = self.model.predict(self.X_test)

        if self.model_type == 'fasttext':
            pred_labels = [label[0] for label in predictions[0]]
        else:#if self.model_type == 'svm':
            pred_labels = predictions

        for pred_label, label in zip(pred_labels, self.y_test):

            if label != self.oos_label:  # measure accuracy
                if pred_label == label:
                    accuracy_correct += 1

                accuracy_out_of += 1
            else:  # measure recall
                if pred_label == label:
                    recall_correct += 1

                recall_out_of += 1

        accuracy = (accuracy_correct / accuracy_out_of) * 100 if accuracy_out_of != 0 else 0
        recall = (recall_correct / recall_out_of) * 100 if recall_out_of != 0 else 0

        return accuracy, recall

    def test_threshold(self, threshold: float):
        accuracy_correct = 0
        accuracy_out_of = 0

        recall_correct = 0
        recall_out_of = 0

        if self.model_type == 'fasttext':
            predictions = self.model.predict(self.X_test)

            pred_labels = [label[0] for label in predictions[0]]
            pred_similarities = [similarity[0] for similarity in predictions[1]]
        else:#if self.model_type == 'svm':
            prediction_probabilities = self.model.predict_proba(self.X_test)  # intent prediction probabilities

            pred_labels = np.argmax(prediction_probabilities, axis=1)  # intent predictions
            pred_similarities = np.take_along_axis(prediction_probabilities, np.expand_dims(pred_labels, axis=-1),
                                                   axis=-1).squeeze(axis=-1)

        for pred_label, pred_similarity, label in zip(pred_labels, pred_similarities, self.y_test):

            if pred_similarity < threshold:
                pred_label = self.oos_label

            if label != self.oos_label:  # measure accuracy
                if pred_label == label:
                    accuracy_correct += 1

                accuracy_out_of += 1
            else:  # measure recall
                if pred_label == label:
                    recall_correct += 1

                recall_out_of += 1

        accuracy = (accuracy_correct / accuracy_out_of) * 100 if accuracy_out_of != 0 else 0
        recall = (recall_correct / recall_out_of) * 100 if recall_out_of != 0 else 0

        return accuracy, recall

    def test_binary(self):
        accuracy_correct = 0
        accuracy_out_of = 0

        recall_correct = 0
        recall_out_of = 0

        for message, label in zip(self.X_test, self.y_test):
            # 1st step - binary classification
            bin_pred = self.bin_model.predict(message)
            bin_pred_label = bin_pred[0]

            if self.model_type == 'fasttext':
                bin_pred_label = bin_pred_label[0]

            if bin_pred_label != self.oos_label:
                # 2nd step - intent classification
                int_pred = self.model.predict(message)
                int_pred_label = int_pred[0]

                if self.model_type == 'fasttext':
                    int_pred_label = int_pred_label[0]

                if int_pred_label == label:
                    accuracy_correct += 1

                accuracy_out_of += 1
            else:
                if bin_pred_label == label:  # here bin_pred_label is always __label__oos
                    recall_correct += 1

                recall_out_of += 1

        accuracy = (accuracy_correct / accuracy_out_of) * 100 if accuracy_out_of != 0 else 0
        recall = (recall_correct / recall_out_of) * 100 if recall_out_of != 0 else 0

        return accuracy, recall
