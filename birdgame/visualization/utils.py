import pandas as pd
import numpy as np
from IPython.display import display


def get_loc_and_scale(density_dict):
    """
    Extract, if present, the location and scale parameters from a density dictionary.
    If the density is a mixture, it selects the component with the highest weight.
    """
    if not density_dict:
        return None, None
    try:
        dist_type = density_dict.get("type")
        if dist_type == "mixture":
            # if mixture: get loc and scale from highest weight distribution
            # Get index of the highest weight
            max_index = max(range(len(density_dict["components"])), key=lambda i: density_dict["components"][i]["weight"])
            density_dict = density_dict["components"][max_index]["density"]
        params = density_dict["params"]
        loc = params.get("loc", params.get("mu", None))
        scale = params.get("scale", params.get("sigma", None))
        return loc, scale
    except Exception as e:
        return None, None
    

def compute_metric_stats(df: pd.DataFrame):
    """Compute and print median, mean and std of df metrics"""
    stats = df.agg(["mean", "median", "std"]).round(3)
    
    for stat_name, values in stats.iterrows():
        print(f"{stat_name.capitalize()}: {values.to_dict()}")

    return stats


def summarize_predictions(store_pred, skip_length=500):
    """
    Generate and print statistical summaries for prediction scores and prediction data.

    Parameters:
        store_pred (list of dict or pd.DataFrame): Stored predictions containing time, loc, scale, etc.
        skip_length (int, optional): Number of initial values to skip (to avoid warm-up bias). Default is 500.
    """
    # Create DataFrame for prediction history (skipping initial warm-up period)
    pred_summary = pd.DataFrame(store_pred[skip_length:]).round(5)

    # Create DataFrame for score history (skipping initial warm-up period)
    scores = pred_summary[["score", "log_score"]]
    stats_summary = compute_metric_stats(scores)

    # Display and summarize prediction data
    print("\nPrediction Data:")
    display(pred_summary.round(5))

    return stats_summary, pred_summary