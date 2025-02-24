import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display
from sktime.forecasting.ets import AutoETS
from sktime.forecasting.base import ForecastingHorizon
from sktime.utils import plotting
from sktime.datasets import load_airline

def visualize_sktime_model(seasonal_period=12):
    # Example dataset
    y = load_airline()

    # Forecasting horizon
    steps = 10  # Predict the next 10 steps
    fh = ForecastingHorizon(np.arange(1, steps + 1), is_relative=True)

    # Initialize and fit the AutoETS forecaster
    forecaster = AutoETS(auto=True, sp=seasonal_period, information_criterion="aic")
    forecaster.fit(y, fh=fh)

    # Make point forecast
    y_pred = forecaster.predict(fh=fh)

    # Make interval forecast
    coverage = 0.9
    y_pred_ints = forecaster.predict_interval(coverage=coverage)

    fig, ax = plotting.plot_series(
        y, y_pred, labels=["y", "y_pred"], pred_interval=y_pred_ints, 
        title=f"AutoETS forecaster (seasonal period = {seasonal_period})", colors=["grey", "red"]
    )
    ax.legend()
    plt.show()

    # Make Variance forecast
    print("Variance estimation:")
    y_pred_variance = forecaster.predict_var(fh=fh)
    display(y_pred_variance)