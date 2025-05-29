import glob
import pickle
from scipy.sparse import hstack , vstack, csr_matrix, load_npz, save_npz
import numpy as np




def load_num_features(path: str = "/home/mlt_ml1/AmazonPricePrediction/data/processed/numerical_features.pkl"):
    with open(path, 'rb') as f:
        return pickle.load(f) #x_num

def load_metadata(path: str = "/home/mlt_ml1/AmazonPricePrediction/data/processed/metadata.pkl"):
    with open(path, "rb") as f:
        metadata = pickle.load(f)
    y = metadata["y"]  
    return metadata, y

def load_text_features(path: str = "/home/mlt_ml1/AmazonPricePrediction/data/processed/text_features_batch_*.pkl"):
    sorted_files = sorted(glob.glob(path))
    return [pickle.load(open(fn, "rb")) for fn in sorted_files]


    # batches = []
    # for file in sorted_files:
    #     with open(file, "rb") as f:
    #         batches.append(pickle.load(f))
    # return batches

def feature_matrix(X_num, text_batches: list, save_path = None):
    X_text = vstack(text_batches, format="csr")
    X = hstack([X_num, X_text], format="csr")
    if save_path:
        save_npz(save_path, X)
    return X


def load_or_assemble(
    num_path="/home/mlt_ml1/AmazonPricePrediction/data/processed/numerical_features.pkl",
    meta_path="/home/mlt_ml1/AmazonPricePrediction/data/processed/metadata.pkl",
    text_pattern="/home/mlt_ml1/AmazonPricePrediction/data/processed/text_features_batch_*.pkl",
    assembled_path="/home/mlt_ml1/AmazonPricePrediction/data/processed/X_all.npz",
    force_rebuild=False,
):
    try:
        if not force_rebuild:
            X = load_npz(assembled_path)
        else:
            raise FileNotFoundError
    except (FileNotFoundError, ValueError):
        X_num = load_num_features(num_path)
        text_batches = load_text_features(text_pattern)
        X = feature_matrix(X_num, text_batches, save_path=assembled_path)
        # sanity check
        total_text_rows = sum(b.shape[0] for b in text_batches)
        assert total_text_rows == X_num.shape[0], (
            f"Total text rows {total_text_rows} ≠ numeric rows {X_num.shape[0]}"
        )

        X = feature_matrix(X_num, text_batches, save_path=assembled_path)
    meta, y = load_metadata(meta_path)
    return X, meta, y


if __name__ == "__main__":


    X,meta, y = load_or_assemble(force_rebuild=False)
    print("---Features matrix shape---")
    print(X.shape)
    print("---Target vector length:---")
    print(len(y))
          