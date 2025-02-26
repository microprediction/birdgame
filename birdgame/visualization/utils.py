import pandas as pd
from IPython.display import display


def compute_metric_stats(df: pd.DataFrame):
    """Compute and print median, mean and std of df metrics"""
    stats = df.agg(["median", "mean", "std"]).round(3)
    
    for stat_name, values in stats.iterrows():
        print(f"{stat_name.capitalize()}: {values.to_dict()}")

    return stats


def summarize_predictions(score_history, store_pred, skip_length=500):
    """
    Generate and print statistical summaries for prediction scores and prediction data.

    Parameters:
        score_history (list): Historical scores to analyze.
        store_pred (list of dict or pd.DataFrame): Stored predictions containing time, loc, scale, etc.
        skip_length (int, optional): Number of initial values to skip (to avoid warm-up bias). Default is 500.
    """
    # Create DataFrame for score history (skipping initial warm-up period)
    scores = pd.DataFrame({"score": score_history[skip_length:]})
    stats_summary = compute_metric_stats(scores)

    # Display and summarize prediction data
    print("\nPrediction Data:")
    pred_summary = pd.DataFrame(store_pred[skip_length:]).round(5)
    display(pred_summary.round(5))

    return stats_summary, pred_summary