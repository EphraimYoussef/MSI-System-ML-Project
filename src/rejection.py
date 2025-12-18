import numpy as np

UNKNOWN_CLASS_ID = 6  # ID for the "Unknown" class

# ----------------------------------------
# SVM rejection function
# ----------------------------------------
def svm_predict_with_rejection(model, X, threshold=0.6):
    """
    Predict with SVM and reject low-confidence samples.

    Args:
        model : trained SVM (probability=True)
        X     : feature vector(s), shape=(n_samples, n_features)
        threshold : minimum probability to accept prediction

    Returns:
        y_pred : np.array of class IDs (0-5) or 6 for Unknown
        y_conf : np.array of max probabilities
    """
    probs = model.predict_proba(X)
    max_probs = np.max(probs, axis=1)
    preds = model.classes_[np.argmax(probs, axis=1)]

    y_pred = np.array([
        UNKNOWN_CLASS_ID if prob < threshold else cls
        for cls, prob in zip(preds, max_probs)
    ])

    return y_pred, max_probs

# ----------------------------------------
# KNN rejection function
# ----------------------------------------
def knn_predict_with_rejection(model, X, threshold=0.6):
    """
    Predict with KNN and reject low-confidence samples.
    Confidence = fraction of neighbors voting for predicted class.
    """
    distances, indices = model.kneighbors(X)
    neighbor_labels = model._y[indices]  # neighbor labels

    y_pred = []
    y_conf = []

    for i in range(len(X)):
        labels, counts = np.unique(neighbor_labels[i], return_counts=True)
        majority_class = labels[np.argmax(counts)]
        confidence = counts.max() / model.n_neighbors

        if confidence < threshold:
            y_pred.append(UNKNOWN_CLASS_ID)
        else:
            y_pred.append(majority_class)

        y_conf.append(confidence)

    return np.array(y_pred), np.array(y_conf)
