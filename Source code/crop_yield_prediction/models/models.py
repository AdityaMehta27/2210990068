"""
Model Definitions
==================
All six ML algorithms from Table III of the paper, with exact hyperparameters.

Models:
  1. Ridge Regression       – linear baseline
  2. Decision Tree          – shallow, regularised
  3. Random Forest          – ensemble bagging
  4. SVR (RBF kernel)       – kernel-based
  5. XGBoost                – gradient boosting (regularised)
  6. Gradient Boosting      – gradient boosting (sklearn)
"""

from sklearn.linear_model    import Ridge
from sklearn.tree            import DecisionTreeRegressor
from sklearn.ensemble        import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm             import SVR

try:
    from xgboost import XGBRegressor
    _XGBOOST_AVAILABLE = True
except ImportError:
    _XGBOOST_AVAILABLE = False
    print("WARNING: xgboost not installed. Install with: pip install xgboost")

RANDOM_SEED = 42


def get_models() -> dict:
    """
    Returns a dict of {model_name: fitted_estimator_instance}.
    All models use random_state=42 where applicable (paper Section IV-B).
    """
    return {
        "Ridge Regression": Ridge(
            alpha=1.0,
        ),

        "Decision Tree": DecisionTreeRegressor(
            max_depth=8,
            min_samples_leaf=10,
            random_state=RANDOM_SEED,
        ),

        "Random Forest": RandomForestRegressor(
            n_estimators=300,
            max_features="sqrt",
            min_samples_leaf=5,
            n_jobs=-1,
            random_state=RANDOM_SEED,
        ),

        "SVR (RBF)": SVR(
            kernel="rbf",
            C=10,
            epsilon=0.05,
        ),

        **({"XGBoost": XGBRegressor(
            n_estimators=400,
            learning_rate=0.05,
            max_depth=6,
            reg_alpha=0.1,       # L1
            reg_lambda=1.0,      # L2
            subsample=0.8,
            colsample_bytree=0.8,
            n_jobs=-1,
            random_state=RANDOM_SEED,
            verbosity=0,
        )} if _XGBOOST_AVAILABLE else {}),

        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=5,
            subsample=0.8,
            random_state=RANDOM_SEED,
        ),
    }


# Models that require scaled (standardised) features
SCALED_MODELS = {"Ridge Regression", "SVR (RBF)"}
