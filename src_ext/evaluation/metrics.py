from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.preprocessing import MultiLabelBinarizer

from src_ext.retrieval.candidate_builder import split_ec_numbers


def compute_multilabel_metrics(true_labels, pred_labels):
    mlb = MultiLabelBinarizer()
    mlb.fit(list(true_labels) + list(pred_labels))
    y_true = mlb.transform(true_labels)
    y_pred = mlb.transform(pred_labels)

    return {
        "precision_micro": float(precision_score(y_true, y_pred, average="micro", zero_division=0)),
        "recall_micro": float(recall_score(y_true, y_pred, average="micro", zero_division=0)),
        "f1_micro": float(f1_score(y_true, y_pred, average="micro", zero_division=0)),
        "subset_accuracy": float(accuracy_score(y_true, y_pred)),
        "num_samples": int(len(true_labels)),
        "num_labels": int(len(mlb.classes_)),
    }


def labels_from_dataframe(df):
    return [split_ec_numbers(value) for value in df["EC number"].tolist()]
