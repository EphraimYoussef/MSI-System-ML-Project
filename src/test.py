import os
import cv2
import numpy as np
import joblib
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D


# Class mapping (REQUIRED OUTPUT FORMAT)
CLASS_MAPPING = {
    "cardboard": 0,
    "glass": 1,
    "metal": 2,
    "paper": 3,
    "plastic": 4,
    "trash": 5,
    "unknown": 6
}


# Load CNN feature extractor (global, loaded once)
_base_model = ResNet50(
    weights="imagenet",
    include_top=False,
    input_shape=(224, 224, 3)
)
_x = GlobalAveragePooling2D()(_base_model.output)
_feature_extractor = Model(_base_model.input, _x)


# Image preprocessing
def _preprocess_image(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))
    img = np.expand_dims(img, axis=0)
    return preprocess_input(img)


# Prediction with rejection
# Supports both SVM and KNN
def _predict_with_rejection(model, X, threshold=0.6):

    # ---- SVM (probability-based rejection) ----
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X)
        max_probs = np.max(probs, axis=1)
        preds = model.predict(X)

        final_preds = []
        for pred, conf in zip(preds, max_probs):
            if conf < threshold:
                final_preds.append("unknown")
            else:
                final_preds.append(pred)

        return final_preds

    # ---- KNN (vote-based rejection) ----
    distances, indices = model.kneighbors(X)
    neighbor_labels = model._y[indices]
    preds = model.predict(X)

    final_preds = []
    for i, pred in enumerate(preds):
        votes = np.sum(neighbor_labels[i] == pred)
        confidence = votes / model.n_neighbors

        if confidence < threshold:
            final_preds.append("unknown")
        else:
            final_preds.append(pred)

    return final_preds


# REQUIRED FUNCTION
def predict(dataFilePath, bestModelPath):
    """
    Parameters:
        dataFilePath (str): Path to folder containing images
        bestModelPath (str): Path to trained model (.pkl)

    Returns:
        list: List of numeric class predictions
    """

    # Load trained model
    model = joblib.load(bestModelPath)

    # Load scaler (must be in same folder as model)
    scaler_path = os.path.join(
        os.path.dirname(bestModelPath),
        "scaler.pkl"
    )
    scaler = joblib.load(scaler_path)

    features = []

    # Load and process images
    for img_name in sorted(os.listdir(dataFilePath)):
        if not img_name.lower().endswith((".jpg", ".png", ".jpeg")):
            continue

        img_path = os.path.join(dataFilePath, img_name)
        img = cv2.imread(img_path)

        if img is None:
            continue

        img = _preprocess_image(img)
        feat = _feature_extractor.predict(img, verbose=0)[0]
        features.append(feat)

    if len(features) == 0:
        return []

    X = np.array(features)
    X = scaler.transform(X)

    # Predict with rejection
    string_preds = _predict_with_rejection(model, X, threshold=0.6)

    # Map string labels to numeric IDs
    numeric_preds = [
        CLASS_MAPPING.get(pred, CLASS_MAPPING["unknown"])
        for pred in string_preds
    ]

    return numeric_preds
