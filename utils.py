import pandas as pd
from sklearn.metrics import (
    confusion_matrix, accuracy_score, precision_score,
    recall_score, f1_score
)
from typing import List, Tuple, Dict


def classification_metrics_report(y_true:List, 
                                  y_pred:List, 
                                  n_words:List[int], 
                                  labels:List[str]=None,
                                  print_report:bool=True,
                                  thresholds: Tuple[int, int]=(7, 12)):
    
    """
    Calcula y devuelve métricas de evaluación para un modelo de clasificación, 
    incluyendo resultados globales, por clase, matriz de confusión y métricas 
    agrupadas según el tamaño de las frases.

    Args:
    - y_true (List[str]): Etiquetas reales.
    - y_pred (List[str]): Etiquetas predichas por el modelo.
    - n_words (List[int]): Número de palabras por frase (para agrupar por tamaño).
    - labels (List[str], opcional): Lista de etiquetas a considerar. Si no se indica, se detectan automáticamente.
    - print_report (bool, opcional): Si es True, muestra el informe por pantalla.
    """    

    if labels is None:
        labels = sorted(set(y_true) | set(y_pred))

    # Global metrics
    global_metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_macro": precision_score(y_true, y_pred, average='macro', zero_division=0),
        "recall_macro": recall_score(y_true, y_pred, average='macro', zero_division=0),
        "f1_macro": f1_score(y_true, y_pred, average='macro', zero_division=0),
    }
    global_metrics_df = pd.DataFrame(global_metrics, index=["value"])

    # Per-class metrics
    per_class = {
        "label": labels,
        "precision": precision_score(y_true, y_pred, labels=labels, average=None, zero_division=0),
        "recall": recall_score(y_true, y_pred, labels=labels, average=None, zero_division=0),
        "f1": f1_score(y_true, y_pred, labels=labels, average=None, zero_division=0),
        "support": [sum(y == label for y in y_true) for label in labels]
    }
    per_class_df = pd.DataFrame(per_class)

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_df = pd.DataFrame(cm, index=labels, columns=labels)

    # Metrics per size group
    size_labels = []
    for n in n_words:
        if n < thresholds[0]:
            size_labels.append("small")
        elif n <= thresholds[1]:
            size_labels.append("medium")
        else:
            size_labels.append("large")

    size_df = pd.DataFrame({"y_true": y_true, "y_pred": y_pred, "size": size_labels})
    size_groups = size_df.groupby("size")

    metrics_per_size = []
    for size, group in size_groups:
        acc = accuracy_score(group["y_true"], group["y_pred"])
        prec = precision_score(group["y_true"], group["y_pred"], average='macro', zero_division=0)
        rec = recall_score(group["y_true"], group["y_pred"], average='macro', zero_division=0)
        f1 = f1_score(group["y_true"], group["y_pred"], average='macro', zero_division=0)
        metrics_per_size.append({
            "size": size,
            "accuracy": acc,
            "precision_macro": prec,
            "recall_macro": rec,
            "f1_macro": f1
        })

    metrics_per_size_df = pd.DataFrame(metrics_per_size)

    if print_report:
        print("=== Global Metrics ===")
        print(global_metrics_df.to_string(index=False))
        print("\n=== Per-Class Metrics ===")
        print(per_class_df.to_string(index=False))
        print("\n=== Metrics Per Size ===")
        print(metrics_per_size_df.to_string(index=False))
        print("\n=== Confusion Matrix ===")
        print(cm_df.to_string())

    return {
        "global_metrics": global_metrics_df,
        "per_class_metrics": per_class_df,
        "metrics_per_size": metrics_per_size_df,
        "confusion_matrix": cm_df
    }


from typing import Tuple

def categorize_length(n: int, thresholds: Tuple[int, int] = (7, 12)) -> str:
    """
    Clasifica una secuencia según su longitud en tres categorías: 'Corta', 'Media' o 'Larga'.

    Args:
    - n (int): Número de palabras de la secuencia.
    - thresholds (Tuple[int, int], opcional): Umbrales que definen los rangos de longitud.
      El primero representa el mínimo de la categoría media (inclusive), y el segundo su máximo (inclusive).
      Por defecto, (7, 12).

    Returns:
    - str: Categoría de longitud: 'Small (<7)', 'Medium (7–12)' o 'Long (>12)'.
    """
    min_med, max_med = thresholds
    if n < min_med:
        return f"Small (<{min_med})"
    elif n <= max_med:
        return f"Medium ({min_med}–{max_med})"
    else:
        return f"Long (>{max_med})"
