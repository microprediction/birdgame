import numpy as np
import matplotlib.pyplot as plt
from ngboost import NGBoost
from sklearn.tree import DecisionTreeRegressor
from ngboost.distns import Normal

# True function and noise
def true_function(X):
    return np.sin(3 * X)

def true_noise_scale(X):
    return np.abs(np.cos(X))

def visualize_ngboost(max_n_estimators=40, step=10, n_data_samples=200):
    """ Function to visualize NGBoost with different number of estimators. """

    # Create synthetic data
    np.random.seed(71)
    X = np.random.uniform(-2, 1, n_data_samples)
    y = true_function(X) + np.random.normal(scale=true_noise_scale(X), size=n_data_samples)

    # Pace points for plotting
    xx = np.linspace(-2.3, 1.3, 300).reshape(-1, 1)

    for i, n in enumerate(range(0, max_n_estimators, step)):
        # NGBoost model
        model = NGBoost(Dist=Normal, learning_rate=0.1, n_estimators=n, natural_gradient=True,
                        verbose=False, random_state=15,
                        validation_fraction=0.1, early_stopping_rounds=None,
                        Base=DecisionTreeRegressor(max_depth=5))
        model.fit(X.reshape(-1, 1), y)

        y_pred_dist = model.pred_dist(xx)

        # Extract the loc and scale of the predictions for each sample
        loc = y_pred_dist.loc
        scale = y_pred_dist.scale
        
        fig, axes = plt.subplots(figsize=(20, 5), nrows=1, ncols=2, sharex=True)
        
        # Plot mean and confidence interval
        ax = axes[0]
        ax.plot(xx, loc, '--', label='Predicted Mean', c='red')
        ax.fill_between(xx.flatten(), loc.flatten() - scale.flatten(), loc.flatten() + scale.flatten(), color='red', label='One Sigma', alpha=0.2)
        
        # Scatter plot of training data and true function
        ax.scatter(X, y, label='Training Data', c='gray')
        ax.plot(xx, true_function(xx), c='gray', label='True Function')
        ax.set_ylim(-3, 3)
        ax.legend(loc=1)
        ax.set_title(f'NGBoost n_estimators={n}')
        
        # Plot uncertainty (scale)
        ax = axes[1]
        ax.set_ylabel('Uncertainty (Scale)')
        ax.plot(xx, scale, label='Predicted Scale', c='red')
        ax.plot(xx, true_noise_scale(xx), '--', label='Ground Truth Noise', c='gray')
        ax.set_ylim(-0.1, np.max(true_noise_scale(xx)) + 0.1)
        ax.legend(loc=1)
        
        fig.tight_layout()
        plt.show()