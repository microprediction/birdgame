import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def plot_dove_predictions(store_pred, start_ind=1000, window_size=200, max_score=100):
    """
    Plots observed vs. predicted dove locations with uncertainty and scores.

    Parameters:
        store_pred (list of dict or DataFrame): Stored predictions as dict format with keys ("time", "loc", "scale", "dove_location", "score").
        start_ind (int): Starting index for slicing data.
        window_size (int): Number of points to plot.
        max_score (float): Maximum value for clipping scores (to prevent extreme values from dominating).
    """
    try:
        # Ensure the data is in DataFrame format
        if isinstance(store_pred, list) or isinstance(store_pred, pd.DataFrame):
            data_slice = pd.DataFrame(store_pred)
        else:
            raise ValueError("Input 'store_pred' must be a list of dicts or a DataFrame.")
        
        required_columns = ["time", "loc", "scale", "dove_location", "score"]
        for col in required_columns:
            if col not in data_slice.columns:
                raise KeyError(f"Missing required column: '{col}'")
        
        # Ensure the slicing indices are within the bounds of the data
        if start_ind > len(store_pred):
            raise IndexError("Slicing indices are out of bounds for the input data.")
            
        end_ind = start_ind + window_size
        data_slice = data_slice.iloc[start_ind:end_ind]

        # time, dove_location, predictions and metrics
        times = data_slice["time"]
        predicted_locs = data_slice["loc"]
        scales = data_slice["scale"]
        dove_location = data_slice["dove_location"]
        scores = data_slice["score"]


        fig, ax1 = plt.subplots(figsize=(10, 5))

        # Plot locations (left y-axis)
        ax1.scatter(times, dove_location, color="grey", label="Observed Dove Location", marker="o", alpha=0.9)
        ax1.plot(times, predicted_locs, label="Predicted Mean (loc)", color="red", linestyle="-")
        ax1.fill_between(times, predicted_locs - scales, predicted_locs + scales, color="red", alpha=0.2, label="±1 Std Dev (Scale)")

        # Left y-axis labels
        ax1.set_xlabel("Time")
        ax1.set_ylabel("Dove Location")
        ax1.legend(loc="upper left")
        ax1.grid(True)

        # Create second right y-axis for metric scores
        ax2 = ax1.twinx()
        ax2.scatter(times, np.clip(scores, 0, max_score), label="Scores", color="green", marker="|", alpha=0.2)

        # Right y-axis labels
        ax2.set_ylabel("Score")
        ax2.legend(loc="upper right")

        plt.title("Observed vs. Predicted Dove Location with Uncertainty and Scores")
        plt.show()

    except ValueError as e:
        print(f"ValueError: {e}")
    except IndexError as e:
        print(f"IndexError: {e}")
    except KeyError as e:
        print(f"KeyError: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == '__main__':
    # store_pred: Stored predictions as dict format with keys ("time", "loc", "scale", "dove_location", "pdf_score")
    store_pred = [
       {'loc': 8397.591025553202, 'scale': 0.009100640109337603, 'dove_location': 8397.59231481859, 'pdf_score': 43.399032388791014, 'time': 365253},
       {'loc': 8397.591025553202, 'scale': 0.009100640109337603, 'dove_location': 8397.59231481859, 'pdf_score': 43.399032388791014, 'time': 365254},
       {'loc': 8397.59231481859, 'scale': 0.009098741896102058, 'dove_location': 8397.59453204784, 'pdf_score': 42.56317074902872, 'time': 365275},
       {'loc': 8397.59231481859, 'scale': 0.009098741896102058, 'dove_location': 8397.596527501462, 'pdf_score': 39.389443816023224, 'time': 365276},
       {'loc': 8397.59231481859, 'scale': 0.009098741896102058, 'dove_location': 8397.596527501462, 'pdf_score': 39.389443816023224, 'time': 365277},
       {'loc': 8397.59231481859, 'scale': 0.009098741896102058, 'dove_location': 8397.588509256848, 'pdf_score': 40.1737368008812, 'time': 365278},
       {'loc': 8397.59231481859, 'scale': 0.009098741896102058, 'dove_location': 8397.581346964864, 'pdf_score': 21.20327205225648, 'time': 365279},
       {'loc': 8397.59453204784, 'scale': 0.009098065702407442, 'dove_location': 8397.581346964864, 'pdf_score': 15.342674482059342, 'time': 365285},
       {'loc': 8397.596527501462, 'scale': 0.009098065702407442, 'dove_location': 8397.581877995288, 'pdf_score': 11.994139472402232, 'time': 365288},
       {'loc': 8397.581346964864, 'scale': 0.009098065702407442, 'dove_location': 8397.58567361975, 'pdf_score': 39.160843937596795, 'time': 365290}
       ]
    
    plot_dove_predictions(store_pred, start_ind=0, window_size=10, max_pdf_score=100)
