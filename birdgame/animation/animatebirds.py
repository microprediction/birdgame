import matplotlib.pyplot as plt
import matplotlib.animation as animation
import itertools
import numpy as np

TIME_WINDOW = 50

def animate_birds(gen, TIME_WINDOW=50.0):
    """
    Animate dove + falcons from a generator, but only show data from
    the last TIME_WINDOW units of time, and force the y-axis to have
    a total length of exactly 2 on each update.

    Assumes each record from the generator has:
      - 'time'            (numeric)
      - 'dove_location'   (scalar)
      - 'falcon_id'       (any hashable ID)
      - 'falcon_location' (scalar)
    """

    fig, ax = plt.subplots(figsize=(10, 6))

    # We'll store all incoming data in lists/dicts
    dove_times = []
    dove_locs = []

    falcon_data = {}
    color_cycle = itertools.cycle(plt.cm.tab10.colors)

    # Create the dove line
    dove_line, = ax.plot([], [], 'm-', label='Dove')

    def update(frame):
        data = next(gen, None)
        if data is None:
            return  # No more data => stop animation

        # Extract fields
        t = data['time']
        dove_loc = data['dove_location']
        fid = data['falcon_id']
        floc = data['falcon_location']

        # Skip out-of-order times if desired
        if dove_times and t < dove_times[-1]:
            print(f"Skipping out-of-order time: {t} < {dove_times[-1]}")
            return

        # Append dove data
        dove_times.append(t)
        dove_locs.append(dove_loc)

        # Falcon data
        if fid not in falcon_data:
            falcon_data[fid] = {
                't': [],
                'loc': [],
                'scatter': ax.scatter([], [],
                                      color=next(color_cycle),
                                      label=f'Falcon {fid}')
            }
        falcon_data[fid]['t'].append(t)
        falcon_data[fid]['loc'].append(floc)

        # Trim old data for fixed time window
        cutoff = t - TIME_WINDOW
        while dove_times and dove_times[0] < cutoff:
            dove_times.pop(0)
            dove_locs.pop(0)

        for f_id, f_dict in falcon_data.items():
            f_times = f_dict['t']
            f_locs  = f_dict['loc']
            while f_times and f_times[0] < cutoff:
                f_times.pop(0)
                f_locs.pop(0)

        # Update dove line
        dove_line.set_data(dove_times, dove_locs)

        # Update each falcon scatter
        for f_id, f_dict in falcon_data.items():
            sc = f_dict['scatter']
            xvals = np.array(f_dict['t'])
            yvals = np.array(f_dict['loc'])
            sc.set_offsets(np.column_stack((xvals, yvals)))

        # Let Matplotlib autoscale first
        ax.relim()
        ax.autoscale_view()

        # Force the y-axis to be exactly 2 units tall
        ymin, ymax = ax.get_ylim()
        if ymin != ymax:
            ycenter = 0.5 * (ymin + ymax)
            ax.set_ylim(ycenter - 0.5, ycenter + 0.5)  # total range = 2

        # Return updated artists
        all_artists = [dove_line] + [f['scatter'] for f in falcon_data.values()]
        return all_artists

    ani = animation.FuncAnimation(fig, update, interval=100, blit=False)

    plt.xlabel("Time")
    plt.ylabel("Location")
    plt.title("Bird Animation (Fixed Time Window, Y-axis length = 2)")
    plt.legend()
    plt.show()


if __name__ == '__main__':
    from birdgame.datasources.remotetestdata import remote_test_data_generator
    gen = remote_test_data_generator()

    animate_birds(gen=gen, TIME_WINDOW=TIME_WINDOW)
