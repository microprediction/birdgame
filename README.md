# birdgame

Utilities for the Bird Game at [crunchdao.com](https://crunchdao.com). Your task is to track the dove location. 

## Install

```bash
pip install birdgame
```

## Visualize the challenge
Run [animatebirds.py](https://github.com/microprediction/birdgame/blob/main/birdgame/animation/animatebirds.py) to get a quick sense. 

![](https://github.com/microprediction/birdgame/blob/main/docs/assets/bird_animation.png)


## Create your Tracker

To create your tracker, you need to define a class that implements the `TrackerBase` interface. Specifically, your class must implement the following methods:

1. **`tick(self, payload)`**  
   This method is called at every time step to process new payloads. Use this method to update your internal state or logic as needed.

   Payload Example:
     ```python
      {
        "falcon_location": 21.179864629354732,
        "time": 230.96231205799998,
        "dove_location": 19.164986723324326,
        "falcon_id": 1
      }
     ```
2. **`predict(self)`**  
   This method should return your prediction of the dove's location at a future time step. Ensure that the return format complies with the [density_pdf](https://github.com/microprediction/densitypdf/blob/main/densitypdf/__init__.py) specification.

You can refer to the [Tracker examples](https://github.com/microprediction/birdgame/tree/main/birdgame/examples) for guidance.

## Challenge your Tracker against the benchmark

To compare your Tracker's performance against the benchmark Tracker, use the `test_run` method provided in the `TrackerBase` class. This method evaluates your Tracker's efficiency over a series of time steps using [density_pdf](https://github.com/microprediction/densitypdf/blob/main/densitypdf/__init__.py) scoring. **A higher score reflects more accurate predictions.**

### Usage Example:
```python
from birdgame.tracker import TrackerBase

class MyTracker(TrackerBase):
    def tick(self, payload):
        # Process the payload and update internal state
        pass
   
    def predict(self):
        # Return the predicted dove location
        pass
   
# Instantiate your Tracker
tracker = MyTracker()

# Run the test to compare against the benchmark Tracker
tracker.test_run(
    live=True, # Set to True to use live streaming data; set to False to use data from a CSV file
    step_print=1000 # Print the score and progress every 1000 steps
)
```


## Tracker examples 
See [Tracker examples](https://github.com/microprediction/birdgame/tree/main/birdgame/examples). There are:

- Quickstarter Notebooks
- Self-contained examples
- Examples that build on provided classes

or [Trackers](https://github.com/microprediction/birdgame/tree/main/birdgame/models) (Self-contained trackers)

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

