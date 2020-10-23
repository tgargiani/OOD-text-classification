import fasttext, os
import numpy as np

incomplete_path = '/Users/tommaso.gargiani/Documents/FEL/PROJ/datasets'

for dataset_size in ['data_full', 'data_small', 'data_imbalanced']:
    print(f'Testing on: {dataset_size}')

    path = os.path.join(incomplete_path, dataset_size, 'fasttext_labels', 'labels.')
    val_true = []  # used to find the validation threshold
    thresholds = np.linspace(0, 1, 101)

    with open(path + 'val_oos_val', 'r') as f:
        raw = f.read()

    for line in raw.splitlines():
        label, message = line.split(' ', 1)
        val_true.append((label, message))

    test_true = []  # used to check correctness of results

    with open(path + 'test_oos_test', 'r') as f:
        raw = f.read()

    for line in raw.splitlines():
        label, message = line.split(' ', 1)
        test_true.append((label, message))

    # model = fasttext.train_supervised(
    #     input=path + 'train', dim=100,
    #     pretrainedVectors='/Users/tommaso.gargiani/Documents/FEL/PROJ/pretrained_vectors/cc.en.100.vec')

    model = fasttext.train_supervised(
        input=path + 'train', dim=300,
        pretrainedVectors='/Users/tommaso.gargiani/Documents/FEL/PROJ/pretrained_vectors/cc.en.300.vec')

    val_predictions_labels = []

    for label, message in val_true:
        pred = model.predict(message)
        val_predictions_labels.append( (pred, label) )

    # Initialize search for best threshold
    previous_val_accuracy = 0
    threshold = 0

    # Find best threshold
    for idx, tr in enumerate(thresholds):
        val_accuracy_correct = 0
        val_accuracy_out_of = 0

        for pred, label in val_predictions_labels:
            pred_label = pred[0][0]
            similarity = pred[1][0]

            if similarity < tr:
                pred_label = '__label__oos'

            if pred_label == label:
                val_accuracy_correct += 1

            val_accuracy_out_of += 1

        val_accuracy = (val_accuracy_correct / val_accuracy_out_of) * 100

        if val_accuracy < previous_val_accuracy:
            threshold = thresholds[idx - 1] # best threshold is the previous one
            break

        previous_val_accuracy = val_accuracy
        threshold = tr

    # Test
    accuracy_correct = 0
    accuracy_out_of = 0

    recall_correct = 0
    recall_out_of = 0

    for label, message in test_true:
        pred = model.predict(message)
        pred_label = pred[0][0]
        similarity = pred[1][0]

        if similarity < threshold:
            pred_label = '__label__oos'

        # print(best_label, similarity, pred, message)

        if label != '__label__oos':  # measure accuracy
            if pred_label == label:
                accuracy_correct += 1

            accuracy_out_of += 1
        else:  # measure recall
            if pred_label == label:
                recall_correct += 1

            recall_out_of += 1

    accuracy = (accuracy_correct / accuracy_out_of) * 100 if accuracy_out_of != 0 else 0
    recall = (recall_correct / recall_out_of) * 100 if recall_out_of != 0 else 0
    print(f'dataset_size: {dataset_size} -- accuracy: {round(accuracy, 1)}, recall: {round(recall, 1)}\n')
