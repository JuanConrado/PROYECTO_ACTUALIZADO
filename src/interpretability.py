"""
Interpretabilidad: importancia de features (XGBoost) + LIME (local).
"""
import numpy as np
import pandas as pd

import lime
import lime.lime_tabular


def xgb_feature_importance(model, feature_names,
                             importance_type: str = "gain",
                             top_k: int = 15) -> pd.DataFrame:
    """
    Devuelve un DataFrame con las top-k features según un tipo de importancia.

    importance_type ∈ {'gain', 'weight', 'cover', 'total_gain', 'total_cover'}.
    """
    booster = (model.get_booster() if hasattr(model, "get_booster")
               else model)
    score_dict = booster.get_score(importance_type=importance_type)

    # Mapeo f0,f1,... → nombres reales
    items = []
    for k, v in score_dict.items():
        idx = int(k.replace("f", ""))
        items.append({"feature": feature_names[idx], "importance": v})

    df = pd.DataFrame(items)
    df = df.sort_values("importance", ascending=False).head(top_k).reset_index(drop=True)
    return df


def lime_explainer(X_train, feature_names, mode: str = "regression"):
    """Crea un LimeTabularExplainer."""
    return lime.lime_tabular.LimeTabularExplainer(
        np.asarray(X_train),
        feature_names=feature_names,
        mode=mode,
        random_state=42,
    )


def explain_with_lime(explainer, instance, predict_fn, num_features: int = 10):
    """Genera la explicación local LIME para una instancia."""
    exp = explainer.explain_instance(
        np.asarray(instance), predict_fn, num_features=num_features
    )
    return exp


def lime_to_dataframe(explanation) -> pd.DataFrame:
    """Convierte la explicación LIME en DataFrame ordenado por impacto."""
    items = explanation.as_list()
    df = pd.DataFrame(items, columns=["feature_condition", "weight"])
    df["abs_weight"] = df["weight"].abs()
    df = df.sort_values("abs_weight", ascending=False).reset_index(drop=True)
    df = df.drop(columns=["abs_weight"])
    return df
