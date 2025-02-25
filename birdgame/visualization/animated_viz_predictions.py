import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from IPython.display import HTML, display, clear_output


def animated_predictions_graph(gen, my_run, bmark_run, window_size=50, use_plt_show=True):
    """
    Generates an animated graph comparing the observed dove location with the predicted locations from two models: 
    'my_run' and 'bmark_run'.

    gen : generator
        A generator that yields new data points (payloads) for updating predictions.
        
    my_run : TrackerEvaluator
        
    bmark_run : TrackerEvaluator
        
    window_size : int, optional, default=50
        The number of data points to display in the window for each frame of the animation. Older data points 
        are discarded once the window exceeds this size.
    """
    # Initialize figure
    fig, ax1 = plt.subplots(figsize=(10, 5))
    fig.tight_layout()
    plt.subplots_adjust(right=0.75)

    # Lists to store incoming data
    times, dove_locations= [], []
    my_predicted_locs, my_scales, my_scores = [], [], []
    bmark_predicted_locs, bmark_scales, bmark_scores = [], [], []

    # Create placeholders for the plot elements
    known_scatter, = ax1.plot([], [], "o", color="green", label="Known Dove Location")
    future_scatter, = ax1.plot([], [], "o", color="grey", label="Future Dove Location")
    my_predicted_line, = ax1.plot([], [], "-", color="red", label="My Predicted")
    bmark_predicted_line, = ax1.plot([], [], "-", color="blue", label="Bmark Predicted")

    # Score legend box
    text_box = ax1.text(1.05, 0.25, "", transform=ax1.transAxes, fontsize=12,
                        bbox=dict(facecolor='white', alpha=0.6))

    ax1.set_xlabel("Time")
    ax1.set_ylabel("Dove Location")
    ax1.legend(loc="upper left")
    ax1.grid(True)
    plt.title("Animated: Observed vs. Predicted Dove Location with Uncertainty and Scores")

    # Initialize uncertainty fill object
    uncertainty_fill, bmark_uncertainty_fill = None, None

    def update(frame):
        """Update function for animation."""
        nonlocal uncertainty_fill, bmark_uncertainty_fill 

        payload = next(gen, None)  # Get next data point
        if payload is None:
            return  # Stop if generator is exhausted
        
        if my_run.tracker.count < 200:
            my_run.tracker.tick(payload)
            bmark_run.tracker.tick(payload)

        # Run prediction and extract values
        my_run.tick_and_predict(payload)
        bmark_run.tick_and_predict(payload)

        current_time = my_run.time
        dove_location = my_run.dove_location

        my_loc = my_run.loc
        my_scale = my_run.scale
        my_score = my_run.score()

        bmark_loc = bmark_run.loc
        bmark_scale = bmark_run.scale
        bmark_score = bmark_run.score()

        if current_time is None or dove_location is None:
            return
        
        if my_loc is None or my_scale is None:
            print("No 'loc' or 'scale' parameter provided by your distribution.")
            return

        # Append new data
        times.append(current_time)
        dove_locations.append(dove_location)

        my_predicted_locs.append(my_loc)
        my_scales.append(my_scale)
        my_scores.append(my_score)

        bmark_predicted_locs.append(bmark_loc)
        bmark_scales.append(bmark_scale)
        bmark_scores.append(bmark_score)
        

        # Keep only last `window_size` points
        times_trimmed = times[-window_size:]
        dove_locations_trimmed = dove_locations[-window_size:]

        my_predicted_locs_trimmed = my_predicted_locs[-window_size:]
        my_scales_trimmed = my_scales[-window_size:]
        my_scores_trimmed = my_scores[-window_size:]

        bmark_predicted_locs_trimmed = bmark_predicted_locs[-window_size:]
        bmark_scales_trimmed = bmark_scales[-window_size:]
        bmark_scores_trimmed = bmark_scores[-window_size:]

        # Update observed scatter plot
        future_scatter.set_data([times_trimmed[i] for i in range(len(times_trimmed)) if times_trimmed[i] >= current_time - my_run.tracker.horizon],
                                [dove_locations_trimmed[i] for i in range(len(times_trimmed)) if times_trimmed[i] >= current_time - my_run.tracker.horizon])
        known_scatter.set_data([times_trimmed[i] for i in range(len(times_trimmed)) if times_trimmed[i] < current_time - my_run.tracker.horizon],
                                [dove_locations_trimmed[i] for i in range(len(times_trimmed)) if times_trimmed[i] < current_time - my_run.tracker.horizon])

        # Update predicted mean line
        my_predicted_line.set_data(times_trimmed, my_predicted_locs_trimmed)
        bmark_predicted_line.set_data(times_trimmed, bmark_predicted_locs_trimmed)

        # Remove previous uncertainty fill (if exists)
        if uncertainty_fill is not None:
            uncertainty_fill.remove()

        # Add new uncertainty fill (My)
        uncertainty_fill = ax1.fill_between(times_trimmed, 
                                            np.array(my_predicted_locs_trimmed) - np.array(my_scales_trimmed), 
                                            np.array(my_predicted_locs_trimmed) + np.array(my_scales_trimmed), 
                                            color="red", alpha=0.2)
        
        if bmark_uncertainty_fill is not None:
            bmark_uncertainty_fill.remove()

        # Add new uncertainty fill (Bmark)
        bmark_uncertainty_fill = ax1.fill_between(times_trimmed, 
                                            np.array(bmark_predicted_locs_trimmed) - np.array(bmark_scales_trimmed), 
                                            np.array(bmark_predicted_locs_trimmed) + np.array(bmark_scales_trimmed), 
                                            color="blue", alpha=0.2)

        if times_trimmed and my_scores_trimmed:
            text_box.set_text(f"My median Score:    {my_scores_trimmed[-1]:.4f}\n"
                            f"Bmark median Score: {bmark_scores_trimmed[-1]:.4f}")

        # Adjust x-axis limits dynamically to center the window
        x_min = times_trimmed[0]
        x_max = times_trimmed[-1]
        ax1.set_xlim(x_min, x_max + 0.3 * (x_max - x_min))  # Adding margin to the right

        ax1.legend(loc="upper left", bbox_to_anchor=(1.05, 1), borderaxespad=0.)

        ax1.relim()
        ax1.autoscale_view()

        # Force a real-time update of the plot
        clear_output(wait=True)
        display(fig)

    ani = FuncAnimation(fig, update, interval=100, blit=False)

    if use_plt_show:
        plt.show()

    return ani