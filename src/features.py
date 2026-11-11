### Features.py
### Includes all functions needed for assessing feature importance

### Needed Imports
import pandas as pd
import matplotlib.pyplot as plt

def basic_feat_importance(model, features, plot=False, show=False, title="Feature Importance For Model"):
    '''Uses .feature_importances_ to give a basic importance value per feature; values sum to 1.

    Each score is the share of total loss (impurity) reduction attributable to that
    feature, summed over all splits and averaged across trees.

    Inputs:
      model: fitted tree-based estimator exposing .feature_importances_
        (e.g. RandomForest, GradientBoosting, XGBoost).
      features: list-like of feature names, in the same order as the columns
        the model was trained on.
      plot: bool, default False. If True, draws a horizontal bar chart.
      show: bool, default False. If True, prints importances rounded to 3 decimals.
      title: str, title of the plot (only used if plot=True).

    Returns:
      pandas.Series of importances indexed by feature name, sorted ascending.
    '''
    importance = pd.Series(model.feature_importances_, index=features).sort_values()
    if plot:
        ax = importance.plot.barh(figsize=(6, 2.5), color="tab:blue")
        ax.set(xlabel="Share of the loss reduction", title=title)
        ax.grid(alpha=0.3, axis="x")

    if show:
        print(importance.round(3).to_string())

    return importance
