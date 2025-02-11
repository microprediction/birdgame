
# A list of potentially useful packages related to online distributional prediction


### River [github](https://github.com/online-ml/river)

River is a Python library for online machine learning. It aims to be the most user-friendly library for doing machine learning on streaming data. River is the result of a merger between creme and scikit-multiflow.




### Precise [github](https://github.com/microprediction/precise)

Contains a collection of *online* (incremental) [covariance forecasting](https://github.com/microprediction/precise/blob/main/LISTING_OF_COV_SKATERS.md) and [portfolio construction](https://github.com/microprediction/precise/blob/main/LISTING_OF_MANAGERS.md) functions. See [docs](https://microprediction.github.io/precise/). The style is 
functional, which might not please everyone.


### Timemachines [github](https://github.com/microprediction/timemachines)

Univariate prediction functions from diverse packages supported in a simple stateless pure function syntax, mosty for benchmarking and application-specific selection purposes. See [basic usage](https://github.com/microprediction/timemachines/blob/main/examples/basic_usage/run_skater.py). Briefly: if `yt` is a list of floats we can feed them one at a time to a skater like so:

     from timemachines.skaters.somepackage.somevariety import something as f
     for yt in y:
         xt, xt_std, s = f(y=yt, s=s, k=3)
         
This emits a k-vector `xt` of forecasts, and corresponding k-vector `xt_std` of estimated standard errors, and the posterior state `s` needed for the next call. See [skaters](https://microprediction.github.io/timemachines/skaters) for choices of `somepackage`, `somevariety` and `something`. You can also ensemble, compose, bootstrap and do other things with one line of code. The `f` is called a `skater`. These are ([documented](https://microprediction.github.io/timemachines/) and [assessed](https://microprediction.github.io/timeseries-elo-ratings/html_leaderboards/overall.html)).  See [why](https://microprediction.github.io/timemachines/why) for motivation for doing things in **walk-forward incremental** functional fashion with **one line of code**. 

### Momentum [github](https://github.com/microprediction/momentum)

A trivial mini-package for computing the running univariate mean, variance, kurtosis and skew. No dependencies ... not even numpy. No classes ... unless you want them. State is a dict, for trivial serialization.
Tested against scipy, creme, statistics. 


# Other interesting packages 

### Pomegranate [github](https://github.com/jmschrei/pomegranate)

pomegranate is a library for probabilistic modeling defined by its modular implementation and treatment of all models as the probability distributions they are. The modular implementation allows one to easily drop normal distributions into a mixture model to create a Gaussian mixture model just as easily as dropping a gamma and a Poisson distribution into a mixture model to create a heterogeneous mixture. But that's not all! Because each model is treated as a probability distribution, Bayesian networks can be dropped into a mixture just as easily as a normal distribution, and hidden Markov models can be dropped into Bayes classifiers to make a classifier over sequences. Together, these two design choices enable a flexibility not seen in any other probabilistic modeling package.

Recently, pomegranate (v1.0.0) was rewritten from the ground up using PyTorch to replace the outdated Cython backend. This rewrite gave me an opportunity to fix many bad design choices that I made as a bb software engineer. Unfortunately, many of these changes are not backwards compatible and will disrupt workflows. On the flip side, these changes have significantly sped up most methods, improved and simplified the code, fixed many issues raised by the community over the years, and made it significantly easier to contribute. I've written more below, but you're likely here now because your code is broken and this is the tl;dr.

