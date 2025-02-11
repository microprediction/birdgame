import matplotlib.pyplot as plt
import matplotlib.animation as animation
import itertools
from collections import deque
import numpy as np


def animate_birds(gen):
    """
    Animate birds
     - Falcons are shown as dots color-coded by their falcon_id
     - Doves are shown as dots connected by a magenta line
     - Time is shown on the x-axis
     - Scroll the plot to show a buffer of 100 observations at a time

    :param gen:  A generator returning dicts with keys  'dove_location', 'falcon_location', 'time', 'falcon_id'
    :return:
    """
    fig, ax = plt.subplots(figsize=(12, 6))  # Make the plot larger
    buffer_size = 100
    time_window = deque(maxlen=buffer_size)
    dove_locations = deque(maxlen=buffer_size)
    falcon_locations = {}  # Dictionary mapping falcon_id to their deque of positions
    colors = {}  # Dictionary to store falcon colors
    color_cycle = itertools.cycle(plt.cm.tab10.colors)  # Color cycle for falcons

    dove_line, = ax.plot([], [], 'm-', label='Dove')  # Magenta line for doves
    falcon_scatters = {}  # Dict to store scatter objects for falcons

    def update(frame):
        data = next(gen, None)
        if data is None:
            return

        time = data['time']
        dove_location = data['dove_location']
        falcon_location = data['falcon_location']
        falcon_id = data['falcon_id']

        # Ensure dove_location is a scalar
        if isinstance(dove_location, (list, np.ndarray)):
            dove_location = np.mean(dove_location)  # Take mean if it's an array

        # Update time and dove locations
        time_window.append(time)
        dove_locations.append(dove_location)

        # Ensure falcon_location is a dictionary (handles multiple falcons per frame)
        if isinstance(falcon_location, dict):
            for fid, loc in falcon_location.items():
                if fid not in falcon_locations:
                    falcon_locations[fid] = deque(maxlen=buffer_size)
                    colors[fid] = next(color_cycle)
                    falcon_scatters[fid] = ax.scatter([], [], color=colors[fid])
                falcon_locations[fid].append(loc)
        else:  # Single falcon case
            if falcon_id not in falcon_locations:
                falcon_locations[falcon_id] = deque(maxlen=buffer_size)
                colors[falcon_id] = next(color_cycle)
                falcon_scatters[falcon_id] = ax.scatter([], [], color=colors[falcon_id])
            falcon_locations[falcon_id].append(falcon_location)

        # Update dove line
        dove_line.set_data(time_window, dove_locations)

        # Update falcon scatter points
        for fid, locations in falcon_locations.items():
            if len(locations) == len(time_window):
                falcon_scatters[fid].set_offsets(np.column_stack((time_window, locations)))

        ax.relim()
        ax.autoscale_view()
        return [dove_line] + list(falcon_scatters.values())

    ani = animation.FuncAnimation(fig, update, interval=100, blit=False)
    plt.xlabel("Time")
    plt.ylabel("Location")
    plt.title("Bird Animation")
    plt.show()


if __name__ == '__main__':
    from birdgame.datasources.remotetestdata import remote_test_data_generator

    gen = remote_test_data_generator()
    animate_birds(gen=gen)
