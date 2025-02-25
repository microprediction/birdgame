# birdgame

Utilities for the Bird Game at [crunchdao.com](https://crunchdao.com). Your task is to predict the dove location. 

## Install

```bash
pip install birdgame
```

## Visualize the challenge
Run [animatebirds.py](https://github.com/microprediction/birdgame/blob/main/birdgame/animation/animatebirds.py) to get a quick sense. 

![](https://github.com/microprediction/birdgame/blob/main/docs/assets/bird_animation.png)

## Tracker examples 
See [examples](https://github.com/microprediction/birdgame/tree/main/birdgame/examples). There are:

- Quickstarter Notebooks
- Self-contained examples
- Examples that build on provided classes

or [models](https://github.com/microprediction/birdgame/tree/main/birdgame/models) (Self-contained models)

Take your pick! 

## General Bird Game Advice 

The Bird Game challenges you to predict the dove's location using probabilistic forecasting.

### Probabilistic Forecasting

Probabilistic forecasting provides **a distribution of possible future values** rather than a single point estimate, allowing for uncertainty quantification. Instead of predicting only the most likely outcome, it estimates a range of potential outcomes along with their probabilities by outputting a **probability distribution**.

A probabilistic forecast models the conditional probability distribution of a future value $(Y_t)$ given past observations $(\mathcal{H}_{t-1})$. This can be expressed as:  

$$P(Y_t \mid \mathcal{H}_{t-1})$$

where $(\mathcal{H}_{t-1})$ represents the historical data up to time $(t-1)$. Instead of a single prediction $(\hat{Y}_t)$, the model estimates a full probability distribution $(f(Y_t \mid \mathcal{H}_{t-1}))$, which can take different parametric forms, such as a Gaussian:

$$Y_t \mid \mathcal{H}_{t-1} \sim \mathcal{N}(\mu_t, \sigma_t^2)$$

where $(\mu_t)$ is the predicted mean and $(\sigma_t^2)$ represents the uncertainty in the forecast.

Probabilistic forecasting can be handled through various approaches, including **variance forecasters**, **quantile forecasters**, **interval forecasters** or **distribution forecasters**, each capturing uncertainty differently.

For example, you can try to forecast the target location by a gaussian density function (or a mixture), thus the model output follows the form:

```python
{"density": {
                "name": "normal",
                "params": {"loc": y_mean, "scale": y_var}
            },
 "weight": weight
}
```

A **mixture density**, such as the gaussion mixture $\sum_{i=1}^{K} w_i \mathcal{N}(Y_t | \mu_i, \sigma_i^2)$ allows for capturing multi-modal distributions and approximate more complex distributions.

![](https://github.com/microprediction/birdgame/blob/main/docs/assets/proba_forecast.png)

### Additional Resources

- [Literature](https://github.com/microprediction/birdgame/blob/main/LITERATURE.md) 
- Useful Python [packages](https://github.com/microprediction/birdgame/blob/main/PACKAGES.md)

