"""
Builders de Pipelines para todos los modelos de ML.
TODOS los modelos viven dentro de un Pipeline para evitar data leakage.
"""
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import (Ridge, Lasso, ElasticNet,
                                  LogisticRegression)
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.svm import SVR, SVC
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

from xgboost import XGBRegressor, XGBClassifier

from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE, ADASYN

from src.config import RANDOM_STATE


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------

def _impute_scale(model, scale=True):
    steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale:
        steps.append(("scaler", StandardScaler()))
    steps.append(("model", model))
    return Pipeline(steps)


# ------------------------------------------------------------
# Regresión
# ------------------------------------------------------------

def get_regression_pipelines() -> dict:
    return {
        "KNN": _impute_scale(KNeighborsRegressor(n_neighbors=11, n_jobs=-1)),
        "Ridge": _impute_scale(Ridge(alpha=1.0, random_state=RANDOM_STATE)),
        "Lasso": _impute_scale(
            Lasso(alpha=0.001, max_iter=10000, random_state=RANDOM_STATE)
        ),
        "ElasticNet": _impute_scale(
            ElasticNet(alpha=0.001, l1_ratio=0.5,
                       max_iter=10000, random_state=RANDOM_STATE)
        ),
        "Decision Tree": _impute_scale(
            DecisionTreeRegressor(max_depth=10, random_state=RANDOM_STATE),
            scale=False
        ),
        "Random Forest": _impute_scale(
            RandomForestRegressor(n_estimators=200, max_depth=10,
                                   random_state=RANDOM_STATE, n_jobs=-1),
            scale=False
        ),
        "SVR": _impute_scale(SVR(kernel="rbf", C=1.0, epsilon=0.001)),
        "XGBoost": _impute_scale(
            XGBRegressor(n_estimators=200, max_depth=5, learning_rate=0.05,
                         random_state=RANDOM_STATE, verbosity=0,
                         tree_method="hist"),
            scale=False
        ),
    }


# ------------------------------------------------------------
# Clasificación
# ------------------------------------------------------------

def get_classification_pipelines(class_weight=None) -> dict:
    """
    class_weight : None, 'balanced', o dict
        Aplicado a los modelos que lo soportan.
    """
    cw = class_weight  # alias

    return {
        "KNN": _impute_scale(KNeighborsClassifier(n_neighbors=11, n_jobs=-1)),
        "LogReg L1": _impute_scale(
            LogisticRegression(penalty="l1", solver="saga",
                               C=1.0, max_iter=5000,
                               random_state=RANDOM_STATE,
                               class_weight=cw, n_jobs=-1)
        ),
        "LogReg L2": _impute_scale(
            LogisticRegression(penalty="l2", solver="lbfgs",
                               C=1.0, max_iter=5000,
                               random_state=RANDOM_STATE,
                               class_weight=cw)
        ),
        "Naive Bayes": _impute_scale(GaussianNB(), scale=False),
        "Decision Tree": _impute_scale(
            DecisionTreeClassifier(max_depth=10,
                                    random_state=RANDOM_STATE,
                                    class_weight=cw),
            scale=False
        ),
        "Random Forest": _impute_scale(
            RandomForestClassifier(n_estimators=200, max_depth=10,
                                    random_state=RANDOM_STATE,
                                    n_jobs=-1, class_weight=cw),
            scale=False
        ),
        "SVM": _impute_scale(
            SVC(kernel="rbf", C=1.0, probability=True,
                random_state=RANDOM_STATE, class_weight=cw)
        ),
        "XGBoost": _impute_scale(
            XGBClassifier(n_estimators=200, max_depth=5, learning_rate=0.05,
                          random_state=RANDOM_STATE, verbosity=0,
                          tree_method="hist", eval_metric="logloss"),
            scale=False
        ),
    }


# ------------------------------------------------------------
# Pipelines con balanceo
# ------------------------------------------------------------

def make_imb_pipeline(model, sampler="smote", scale=True):
    """
    Crea un ImbPipeline: Imputer → [Scaler] → Sampler → Model.
    El sampler se aplica solo a train durante CV (anti-leakage automático).
    """
    steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale:
        steps.append(("scaler", StandardScaler()))

    if sampler == "smote":
        steps.append(("sampler", SMOTE(random_state=RANDOM_STATE)))
    elif sampler == "adasyn":
        steps.append(("sampler", ADASYN(random_state=RANDOM_STATE)))
    elif sampler is None or sampler == "none":
        pass
    else:
        raise ValueError(f"Sampler desconocido: {sampler}")

    steps.append(("model", model))
    return ImbPipeline(steps)


def get_balancing_variants(base_model_factory) -> dict:
    """
    Genera 4 variantes de un mismo modelo:
      - Sin balanceo
      - SMOTE
      - ADASYN
      - class_weight='balanced'
    """
    return {
        "Sin balanceo": _impute_scale(base_model_factory(class_weight=None)),
        "SMOTE": make_imb_pipeline(base_model_factory(class_weight=None),
                                    sampler="smote"),
        "ADASYN": make_imb_pipeline(base_model_factory(class_weight=None),
                                     sampler="adasyn"),
        "class_weight=balanced": _impute_scale(
            base_model_factory(class_weight="balanced")
        ),
    }
