import matplotlib.pyplot as plt
import numpy as np
from collections import deque
from itertools import islice
from matplotlib.animation import FuncAnimation
from IPython.display import HTML, display, clear_output


def animated_predictions_graph(gen, my_run, bmark_run, n_data_points=50, interval_animation=100, use_plt_show=True):
    """
    Generates an animated graph comparing the observed dove location with the predicted locations from two models: 
    'my_run' and 'bmark_run'.

    gen : generator
        A generator that yields new data points (payloads) for updating predictions.
        
    my_run : TrackerEvaluator
        
    bmark_run : TrackerEvaluator
        
    n_data_points : int, optional, default=50
        The number of data points to display in the window for each frame of the animation. Older data points 
        are discarded once the window exceeds this size.
    interval_animation: int, optional, default=100
        Delay between frames in milliseconds. Does not work in a notebook.
    """
    # Initialize figure
    fig, ax1 = plt.subplots(figsize=(10, 5))
    fig.tight_layout()
    plt.subplots_adjust(right=0.75)

    # Lists to store incoming data
    max_len = n_data_points * 2
    times, dove_locations= deque(maxlen=max_len), deque(maxlen=max_len)
    my_predicted_locs, my_scales, my_overall_scores, my_recent_scores = deque(maxlen=max_len), deque(maxlen=max_len), deque(maxlen=max_len), deque(maxlen=max_len)
    bmark_predicted_locs, bmark_scales, bmark_overall_scores, bmark_recent_scores = deque(maxlen=max_len), deque(maxlen=max_len), deque(maxlen=max_len), deque(maxlen=max_len)

    # Create placeholders for the plot elements
    known_scatter, = ax1.plot([], [], "o", color="green", label="Known Dove Location")
    future_scatter, = ax1.plot([], [], "o", color="grey", label="Future Dove Location")
    my_predicted_line, = ax1.plot([], [], "-", color="red", label="My Predicted")
    bmark_predicted_line, = ax1.plot([], [], "-", color="blue", label="Bmark Predicted")

    # Score legend box
    overall_score_text_box = ax1.text(1.05, 0.30, "", transform=ax1.transAxes, fontsize=12,
                                    bbox=dict(facecolor='white', alpha=0.6))
    recent_score_text_box = ax1.text(1.05, 0.10, "", transform=ax1.transAxes, fontsize=12,
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
        
        # if my_run.tracker.count < 200:
        #     my_run.tracker.tick(payload)
        #     bmark_run.tracker.tick(payload)
        #     return

        # Run prediction and extract values
        my_run.tick_and_predict(payload)
        bmark_run.tick_and_predict(payload)

        current_time = my_run.time
        dove_location = my_run.dove_location

        my_loc = my_run.loc
        my_scale = my_run.scale
        my_overall_score = my_run.overall_median_score()
        my_recent_score = my_run.recent_median_score()

        bmark_loc = bmark_run.loc
        bmark_scale = bmark_run.scale
        bmark_overall_score = bmark_run.overall_median_score()
        bmark_recent_score = bmark_run.recent_median_score()

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
        my_overall_scores.append(my_overall_score)
        my_recent_scores.append(my_recent_score)

        bmark_predicted_locs.append(bmark_loc)
        bmark_scales.append(bmark_scale)
        bmark_overall_scores.append(bmark_overall_score)
        bmark_recent_scores.append(bmark_recent_score)
        

        ### Keep only last 2 * `n_data_points` points
        ### The first `n_data_points` points represents the already predicted/known values (the first half)
        ### The last `n_data_points` points represents the future values (the second half)
        # Keep the 2 * `n_data_points` points for known and future dove locations
        times_trimmed_with_future = times
        dove_locations_trimmed_with_future = dove_locations

        # For predictions values, keep points between [-2*n_data_points:-n_data_points] (the first half)
        last_known_points = n_data_points if len(times) >= max_len else None # in the beginning, we don't plot future data above horizon
        times_trimmed = list(islice(times, 0, last_known_points))
        my_predicted_locs_trimmed = list(islice(my_predicted_locs, 0, last_known_points))
        my_scales_trimmed = list(islice(my_scales, 0, last_known_points))
        my_overall_score = my_overall_scores[last_known_points] if last_known_points else my_overall_scores[-1]
        my_recent_score = my_recent_scores[last_known_points] if last_known_points else my_recent_scores[-1]

        bmark_predicted_locs_trimmed = list(islice(bmark_predicted_locs, 0, last_known_points))
        bmark_scales_trimmed = list(islice(bmark_scales, 0, last_known_points))
        bmark_overall_score = bmark_overall_scores[last_known_points] if last_known_points else bmark_overall_scores[-1]
        bmark_recent_score = bmark_recent_scores[last_known_points] if last_known_points else bmark_recent_scores[-1]
        
        current_time = times_trimmed[-1]

        # Update observed scatter plot
        future_scatter.set_data([times_trimmed_with_future[i] for i in range(len(times_trimmed_with_future)) if times_trimmed_with_future[i] >= current_time - my_run.tracker.horizon],
                                [dove_locations_trimmed_with_future[i] for i in range(len(times_trimmed_with_future)) if times_trimmed_with_future[i] >= current_time - my_run.tracker.horizon])
        known_scatter.set_data([times_trimmed_with_future[i] for i in range(len(times_trimmed_with_future)) if times_trimmed_with_future[i] < current_time - my_run.tracker.horizon],
                                [dove_locations_trimmed_with_future[i] for i in range(len(times_trimmed_with_future)) if times_trimmed_with_future[i] < current_time - my_run.tracker.horizon])

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

        if times_trimmed and my_overall_score is not None:
            overall_score_text_box.set_text(f"Overall:\n"
                                            f"My median Score:    {my_overall_score:.4f}\n"
                                            f"Bmark median Score: {bmark_overall_score:.4f}")
            recent_score_text_box.set_text(f"Recent {my_run.score_window_size} data:\n"
                                            f"My median Score:    {my_recent_score:.4f}\n"
                                            f"Bmark median Score: {bmark_recent_score:.4f}")

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

    ani = FuncAnimation(fig, update, interval=interval_animation, blit=False)

    if use_plt_show:
        plt.show()

    return ani