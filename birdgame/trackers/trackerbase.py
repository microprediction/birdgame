
class TrackerBase:
    """
    Base class that handles quarantining of data points before they are eligible for processing.
    """
    def __init__(self, horizon: int):
        self.horizon = horizon
        self.quarantine = []
        self.count = 0

    def add_to_quarantine(self, time, value):
        """ Adds a new value to the quarantine list. """
        self.quarantine.append((time + self.horizon, value))

    def pop_from_quarantine(self, current_time):
        """ Returns the most recent valid data point from quarantine. """
        valid = [(j, (ti, xi)) for (j, (ti, xi)) in enumerate(self.quarantine) if ti <= current_time]
        if valid:
            prev_ndx, (ti, prev_x) = valid[-1]
            self.quarantine = self.quarantine[:prev_ndx]  # Trim the quarantine list
            return prev_x
        return None
