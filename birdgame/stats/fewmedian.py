from river import stats
from collections import deque
import bisect


class FEWMedian(stats.base.Univariate):
    """
    A regular sliding-window median filter.
    
    Stores up to 'window_size' of the most recent data points, 
    and returns the median of that window when .get() is called.
    """

    def __init__(self, window_size=6):
        """
        Args:
            window_size (int): Number of recent samples to keep 
                               for median calculation.
        """
        self.window_size = window_size

        # We'll keep both:
        # 1) A deque for easy popping of old samples
        # 2) A sorted list for efficient median computation
        self._window = deque()
        self._sorted_window = []

    def update(self, x):
        """
        Incorporates a new data point x into the sliding window 
        and returns 'self' (River convention).
        """
        # 1) Add x to the right of the _window
        self._window.append(x)

        # 2) Insert x into the _sorted_window (kept in sorted order via bisect)
        bisect.insort(self._sorted_window, x)

        # 3) If we've exceeded window_size, remove the oldest element from both structures
        if len(self._window) > self.window_size:
            oldest = self._window.popleft()
            # Remove 'oldest' from the sorted list
            idx = bisect.bisect_left(self._sorted_window, oldest)
            self._sorted_window.pop(idx)

        return self

    def tick(self, x):
        """Alias for update(), if you prefer that naming."""
        return self.update(x)

    def get(self):
        """
        Returns the median of the current window. 
        If there are no samples yet, returns 0 by convention.
        """
        n = len(self._sorted_window)
        if n == 0:
            return 0
        mid = n // 2
        if n % 2 == 1:
            return self._sorted_window[mid]
        else:
            # Average of the two middle values
            return 0.5 * (self._sorted_window[mid - 1] + self._sorted_window[mid])

    def to_dict(self):
        """
        Serializes the state of the FEWMedian object to a dictionary.
        Note: we'll store just the raw data for demonstration.
        """
        return {
            'window_size': self.window_size,
            'window': list(self._window),
            'sorted_window': self._sorted_window
        }

    @classmethod
    def from_dict(cls, data):
        """
        Deserializes the state from a dictionary into a new FEWMedian instance.
        """
        instance = cls(window_size=data['window_size'])
        instance._window = deque(data['window'])
        instance._sorted_window = data['sorted_window']
        return instance


if __name__ == "__main__":
    # Example usage:
    data_stream = [10, 11, 12, 100, 11, 13, 200, 14, 9, 15, 16, 300, 10]

    # Create a FEWMedian with a window_size of 3
    median_filter = FEWMedian(window_size=3)

    for x in data_stream:
        median_filter.update(x)
        print(f"Input: {x:4}, Median: {median_filter.get():.1f}")
