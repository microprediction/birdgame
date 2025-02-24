import numpy as np
import matplotlib.pyplot as plt
from river import linear_model, preprocessing, optim

def visualize_quantile_regression(lr=0.05):
    """ Function to visualize quantile regression with different quantiles. """

    # Generate synthetic data
    np.random.seed(42)
    X = np.linspace(0, 10, 200)
    y = np.sin(X) + np.random.normal(0, 0.3, size=X.shape)  # True function + noise

    # Initialize models for different quantiles
    models = {}
    quantiles = [0.05, 0.5, 0.95]

    for alpha in quantiles:
        scale = preprocessing.StandardScaler()
        learn = linear_model.LinearRegression(
            intercept_lr=0,
            optimizer=optim.SGD(lr),
            loss=optim.losses.Quantile(alpha=alpha)
        )
        models[f"q {alpha:.2f}"] = preprocessing.TargetStandardScaler(regressor=scale | learn)

    # Make predictions and Train Quantile Regression models
    predictions = {q: [] for q in models.keys()}
    for x_val, y_val in zip(X, y):
        x_dict = {"feature": x_val}
        for q in models.keys():
            pred = models[q].predict_one(x_dict)
            predictions[q].append(pred)
            models[q].learn_one(x_dict, y_val)

    plt.figure(figsize=(10, 5))
    plt.scatter(X, y, label="Observations", alpha=0.5, color="gray")
    plt.plot(X, predictions["q 0.50"], label="Median (q=0.50)", color="blue")
    plt.plot(X, predictions["q 0.05"], label="Lower Bound (q=0.05)", color="red", linestyle="--")
    plt.plot(X, predictions["q 0.95"], label="Upper Bound (q=0.95)", color="green", linestyle="--")
    plt.fill_between(X, predictions["q 0.05"], predictions["q 0.95"], alpha=0.1, label="Predicted 90% interval")

    plt.xlabel("X")
    plt.ylabel("y")
    plt.legend()
    plt.title(f"Quantile Regression - stream learning (lr={lr})")
    plt.show()