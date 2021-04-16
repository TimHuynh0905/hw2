import numpy as np

def confusion_matrix(actual, predictions):
    """
    Given predictions (an N-length numpy vector) and actual labels (an N-length 
    numpy vector), compute the confusion matrix. The confusion 
    matrix for a binary classifier would be a 2x2 matrix as follows:

    [
        [true_negatives, false_positives],
        [false_negatives, true_positives]
    ]

    YOU DO NOT NEED TO IMPLEMENT CONFUSION MATRICES THAT ARE FOR MORE THAN TWO 
    CLASSES (binary).
    
    Compute and return the confusion matrix.

    Args:
        actual (np.array): predicted labels of length N
        predictions (np.array): predicted labels of length N

    Output:
        confusion_matrix (np.array): 2x2 confusion matrix between predicted and actual labels

    """

    print(predictions.shape)
    print(actual.shape)

    if predictions.shape[0] != actual.shape[0]:
        raise ValueError("predictions and actual must be the same length!")

    classes = np.sort(np.unique(actual))
    matrix = np.zeros((2, 2))

    for i in range(2):
        for j in range(2):
            matrix[i, j] = np.sum((actual == classes[i]) & (predictions == classes[j]))
    
    return matrix

def accuracy(actual, predictions):
    """
    Given predictions (an N-length numpy vector) and actual labels (an N-length 
    numpy vector), compute the accuracy:

    Hint: implement and use the confusion_matrix function!

    Args:
        actual (np.array): predicted labels of length N
        predictions (np.array): predicted labels of length N

    Output:
        accuracy (float): accuracy
    """
    if predictions.shape[0] != actual.shape[0]:
        raise ValueError("predictions and actual must be the same length!")

    matrix = confusion_matrix(actual, predictions)

    return (matrix[0, 0] + matrix[1, 1]) / np.sum(matrix) if np.sum(matrix) != 0 else 0

def precision_and_recall(actual, predictions):
    """
    Given predictions (an N-length numpy vector) and actual labels (an N-length 
    numpy vector), compute the precision and recall:

    https://en.wikipedia.org/wiki/Precision_and_recall

    Hint: implement and use the confusion_matrix function!

    Args:
        actual (np.array): predicted labels of length N
        predictions (np.array): predicted labels of length N

    Output:
        precision (float): precision
        recall (float): recall
    """
    if predictions.shape[0] != actual.shape[0]:
        raise ValueError("predictions and actual must be the same length!")

    matrix = confusion_matrix(actual, predictions)

    precision = matrix[1, 1] / (matrix[0, 1] + matrix[1, 1]) if (matrix[0, 1] + matrix[1, 1]) != 0 else 0
    recall = matrix[1, 1] / (matrix[1, 0] + matrix[1, 1]) if (matrix[1, 0] + matrix[1, 1]) != 0 else 0

    return precision, recall
    

def f1_measure(actual, predictions):
    """
    Given predictions (an N-length numpy vector) and actual labels (an N-length 
    numpy vector), compute the F1-measure:

   https://en.wikipedia.org/wiki/Precision_and_recall#F-measure

    Hint: implement and use the precision_and_recall function!

    Args:
        actual (np.array): predicted labels of length N
        predictions (np.array): predicted labels of length N

    Output:
        f1_measure (float): F1 measure of dataset (harmonic mean of precision and 
        recall)
    """
    if predictions.shape[0] != actual.shape[0]:
        raise ValueError("predictions and actual must be the same length!")

    precision, recall = precision_and_recall(actual, predictions)

    return (2 * precision * recall) / (precision + recall) if (precision + recall) != 0 else 0

