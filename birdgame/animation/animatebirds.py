import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.ticker as ticker
import itertools
import numpy as np

TIME_WINDOW = 500  # or 500, whichever you like
INTERVAL = 100
HEIGHT = 0.7
START_TIME = 90000

def animate_birds(gen, TIME_WINDOW=50.0):
    """
    Animate dove + falcons from a generator, but only show data from
    the last TIME_WINDOW units of time, and force the x-axis to be
    [t - TIME_WINDOW, t]. The y-axis is then set to include all data
    in that window with a minimum 2.0 total range.
    """

    fig, ax = plt.subplots(figsize=(10, 6))

    # Force x-axis ticks to be integers (no scientific notation)
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax.ticklabel_format(useOffset=False, style='plain', axis='x')

    dove_times = []
    dove_locs = []

    falcon_data = {}
    color_cycle = itertools.cycle(plt.cm.tab10.colors)

    # Create the dove line
    dove_line, = ax.plot([], [], 'm-', label='Dove')

    def update(frame):
        data = next(gen, None)
        if data is None:
            # No more data => stop animation
            return

        t = data['time']
        dove_loc = data['dove_location']
        fid = data['falcon_id']
        floc = data['falcon_location']

        # Optional: Skip out-of-order times
        if dove_times and t < dove_times[-1]:
            print(f"Skipping out-of-order time: {t} < {dove_times[-1]}")
            return

        # 1) Append new dove data
        dove_times.append(t)
        dove_locs.append(dove_loc)

        # 2) Append new falcon data
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

        # 3) Trim old data outside the last TIME_WINDOW
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

        # 4) Update dove line
        dove_line.set_data(dove_times, dove_locs)

        # 5) Update each falcon scatter
        for f_id, f_dict in falcon_data.items():
            sc = f_dict['scatter']
            xvals = np.array(f_dict['t'])
            yvals = np.array(f_dict['loc'])
            sc.set_offsets(np.column_stack((xvals, yvals)))

        # 6) Force the x-limits to show [t - TIME_WINDOW, t]
        #    (rolling window in time)
        ax.set_xlim(t - TIME_WINDOW, t)

        # 7) Custom manual y-limits: always include min/max + a minimum range
        all_locs = []
        all_locs.extend(dove_locs)
        for f_dict in falcon_data.values():
            all_locs.extend(f_dict['loc'])

        if len(all_locs) > 0:
            y_min = np.min(all_locs)
            y_max = np.max(all_locs)
            data_range = y_max - y_min
            forced_range = HEIGHT  # enforce at least this total range

            if data_range == 0:
                # If all points are the same, make a small range around that point
                center = y_min
                half = forced_range / 2
                ax.set_ylim(center - half, center + half)
            else:
                # If data span < forced_range, enforce forced_range
                if data_range < forced_range:
                    center = 0.5 * (y_min + y_max)
                    half = forced_range / 2
                    ax.set_ylim(center - half, center + half)
                else:
                    # If data span > forced_range, add a small margin
                    margin = 0.05 * data_range
                    ax.set_ylim(y_min - margin, y_max + margin)

        # Return all updated artists
        return [dove_line] + [f['scatter'] for f in falcon_data.values()]

    # Create animation
    ani = animation.FuncAnimation(fig, update, interval=INTERVAL, blit=False)

    plt.xlabel("Time")
    plt.ylabel("Location")
    plt.title("Bird Animation")
    plt.legend()
    plt.show()


if __name__ == '__main__':
    from birdgame.datasources.remotetestdata import remote_test_data_generator
    gen = remote_test_data_generator()
    animate_birds(gen=gen, TIME_WINDOW=TIME_WINDOW)
