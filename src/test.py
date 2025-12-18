import os
from typing import List, Optional, Sequence, Tuple

import cv2
import joblib
import numpy as np
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.models import Model


IMAGE_EXTS: Tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")


def _collect_image_paths(root_dir: str, recursive: bool = True) -> List[str]:
    """
    Collect image file paths from a directory.

    If recursive=True, walks subdirectories too (recommended for typical datasets).
    Returns a sorted list of full file paths.
    """
    if not os.path.isdir(root_dir):
        raise NotADirectoryError(f"dataFilePath is not a directory: {root_dir}")

    paths: List[str] = []

    if recursive:
        for dirpath, _, filenames in os.walk(root_dir):
            for name in filenames:
                if name.lower().endswith(IMAGE_EXTS):
                    paths.append(os.path.join(dirpath, name))
    else:
        for name in os.listdir(root_dir):
            full = os.path.join(root_dir, name)
            if os.path.isfile(full) and name.lower().endswith(IMAGE_EXTS):
                paths.append(full)

    paths.sort()
    return paths


def _get_feature_extractor() -> Model:
    """
    Lazy-cache the CNN feature extractor (so repeated calls are fast).
    """
    if not hasattr(_get_feature_extractor, "_model"):
        base_model = ResNet50(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
        x = GlobalAveragePooling2D()(base_model.output)
        _get_feature_extractor._model = Model(base_model.input, x)  # type: ignore[attr-defined]
    return _get_feature_extractor._model  # type: ignore[attr-defined]


def predict(dataFilePath: str, bestModelPath: str) -> List[int]:
    """
    Parameters:
        dataFilePath (str): Path to a folder that contains images (directly OR in subfolders).
        bestModelPath (str): Path to a trained model file (.pkl).

    Returns:
        list[int]: Numeric class predictions (one per image, in sorted file-path order).
    """
    class_mapping = {
        "cardboard": 0,
        "glass": 1,
        "metal": 2,
        "paper": 3,
        "plastic": 4,
        "trash": 5,
        "unknown": 6,
    }

    if not os.path.isfile(bestModelPath):
        raise FileNotFoundError(f"bestModelPath does not exist or is not a file: {bestModelPath}")

    # --- load trained classifier ---
    model = joblib.load(bestModelPath)

    # --- load scaler if available (same directory as model) ---
    scaler: Optional[object] = None
    scaler_path = os.path.join(os.path.dirname(bestModelPath), "scaler.pkl")
    if os.path.exists(scaler_path):
        scaler = joblib.load(scaler_path)

    # --- collect images (RECURSIVE to support dataset/class subfolders) ---
    image_paths = _collect_image_paths(dataFilePath, recursive=True)
    if not image_paths:
        raise FileNotFoundError(
            f"No images found under: {dataFilePath}\n"
            f"Supported extensions: {', '.join(IMAGE_EXTS)}"
        )

    feature_extractor = _get_feature_extractor()

    # --- preprocess -> CNN features ---
    feats: List[np.ndarray] = []
    unreadable: List[str] = []

    for fpath in image_paths:
        img = cv2.imread(fpath)
        if img is None:
            unreadable.append(fpath)
            continue

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_AREA)
        img = np.expand_dims(img, axis=0).astype(np.float32)
        img = preprocess_input(img)

        feat = feature_extractor.predict(img, verbose=0)[0]  # (2048,)
        feats.append(feat)

    if not feats:
        raise RuntimeError(
            "All images failed to load (cv2.imread returned None).\n"
            f"First few failing paths: {unreadable[:5]}"
        )

    X = np.asarray(feats, dtype=np.float32)
    if scaler is not None:
        # scaler is expected to be an sklearn-like transformer
        X = scaler.transform(X)

    # --- inference + (optional) rejection ---
    threshold = 0.6

    def _to_int_label(pred) -> int:
        if isinstance(pred, (np.integer, int)):
            return int(pred)
        return int(class_mapping.get(str(pred), class_mapping["unknown"]))

    # Case 1: models with predict_proba (e.g., SVC(probability=True), LogisticRegression, etc.)
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X)
        max_probs = np.max(probs, axis=1)
        raw_preds = model.predict(X)

        out: List[int] = []
        for pred, conf in zip(raw_preds, max_probs):
            out.append(class_mapping["unknown"] if float(conf) < threshold else _to_int_label(pred))
        return out

    # Case 2: KNN-like models: vote confidence
    if hasattr(model, "kneighbors") and hasattr(model, "n_neighbors"):
        distances, indices = model.kneighbors(X)
        # sklearn internal labels storage differs by estimator; keep best-effort but robust:
        y = getattr(model, "_y", None)
        if y is None:
            raw_preds = model.predict(X)
            return [_to_int_label(p) for p in raw_preds]

        neighbor_labels = y[indices]
        raw_preds = model.predict(X)

        out: List[int] = []
        for i, pred in enumerate(raw_preds):
            votes = np.sum(neighbor_labels[i] == pred)
            conf = float(votes) / float(model.n_neighbors)
            out.append(class_mapping["unknown"] if conf < threshold else _to_int_label(pred))
        return out

    # Fallback: plain predict
    raw_preds = model.predict(X)
    return [_to_int_label(p) for p in raw_preds]


if __name__ == "__main__":
    dataFilePath = r"..\test"
    bestModelPath = r"..\models\svm_model.pkl"

    preds = predict(dataFilePath, bestModelPath)
    print(preds)
