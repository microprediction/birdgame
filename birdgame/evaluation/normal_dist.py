import math

def compute_pdf_score(observed_dove_location, loc, scale):
    """
    Compute the probability density function (PDF) score for a normal distribution.

    Parameters:
        observed_dove_location (float): The observed location of the dove.
        loc (float): Mean (center) of the normal distribution.
        scale (float): Standard deviation of the normal distribution.

    Returns:
        float: The computed PDF score.
    """
    if scale <= 0:
        raise ValueError("Scale (standard deviation) must be positive.")

    # Compute normalization factor
    normalization_factor = 1 / (math.sqrt(2 * math.pi * scale**2))

    # Compute the decay factor (determines how unlikely the observed value is)
    decay_factor = -((observed_dove_location - loc) ** 2) / (2 * scale**2)

    pdf_score = normalization_factor * math.exp(decay_factor)

    return pdf_score


def compute_squared_z_score(observed_dove_location, loc, scale):
    """Compute the squared Z-score, which measures the deviation from the mean in standard deviation units."""
    if scale <= 0:
        raise ValueError("Scale (standard deviation) must be positive.")

    return ((observed_dove_location - loc) / scale) ** 2


def compute_score_for_normal_dist(pdf, observed_dove_location, metric="pdf_score"):
    """
    Compute a score for a given observed dove location.
    
    The function calculates the score by considering a (mixture of) normal distributions, where 
    each distribution has a weight, mean (loc), and scale (standard deviation). The weighted score 
    is a combination of the individual scores for each component distribution.

    Parameters:
        pdf (dict): A dictionary representing a probability distribution. It contains 
                         a list of components, where each component includes:
                         - 'density': A dictionary with 'loc' (mean) and 'scale' (standard deviation) 
                           for the normal distribution.
                         - 'weight': The weight assigned to this component in the mixture.
        observed_dove_location (float): The observed location of the dove for which the PDF score is 
                                        being calculated.
        metric (str or callable, optional): The metric used to compute the score.
                                            - If a string, must be one of:
                                              - "pdf_score" (default): Uses the standard PDF.
                                              - "z_score": Measures deviation from the mean in standard deviation units.
                                            - If a callable function, it must accept (observed_dove_location, loc, scale)
                                              and return a numeric score.

    Returns:
        - weighted_pdf_score (float)
        - stored_prediction (dict): A dictionary with the predicted parameters (loc, scale), dove location and 
                                    the score corresponding to the component with the highest weight.

    Notes: The function assumes that the score is computed for a normal distribution (Gaussian).

    Example:
        pdf = {
            "components": [
                {"density": {"params": {"loc": 0, "scale": 1}}, "weight": 0.7},
                {"density": {"params": {"loc": 5, "scale": 2}}, "weight": 0.3}
            ]
        }

        # Using predefined metric
        compute_score_for_normal_dist(pdf, observed_dove_location=2, metric="z_score")

        # Using a custom metric function
        def my_custom_metric(obs, loc, scale):
            return abs(obs - loc) / (scale + 1)

        compute_score_for_normal_dist(pdf, observed_dove_location=2, metric=my_custom_metric)
    """
    metric_functions = {
        "pdf_score": compute_pdf_score,
        "z_score": compute_squared_z_score
    }

    weighted_score = 0
    highest_weight = 0
    stored_prediction = None

    # Iterate over each normal distribution
    for component in pdf.get('components', []):
        density = component['density']
        loc, scale, weight = density['params']['loc'], density['params']['scale'], component['weight']

        # Determine which scoring function to use
        if isinstance(metric, str):
            if metric not in metric_functions:
                raise ValueError(f"Unsupported metric: {metric}")
            score_function = metric_functions[metric]
        elif callable(metric):
            score_function = metric  # Custom function
        else:
            raise ValueError("Metric must be a string or a callable function.")
        
        # Compute the score
        score = score_function(observed_dove_location, loc, scale)

        weighted_score += weight * score

        # Store the prediction corresponding to the component with the highest weight
        if weight > highest_weight:
            stored_prediction = {
                    "loc": loc,
                    "scale": scale,
                    "dove_location": observed_dove_location,
                    "score": score,
                }
            highest_weight = weight

    return weighted_score, stored_prediction
