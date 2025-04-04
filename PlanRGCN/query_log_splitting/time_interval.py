import scipy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter


class TimeIntervalExtractorBase:
    def __init__(self, run_times: list[float]):
        self.runtimes = run_times
        self.thresholds = self.create_interval_threshold()

    def create_interval_threshold(self) -> list[float]:
        raise NotImplementedError

    def find_intervals(self):
        interval_counts = []

        for value in self.runtimes:
            interval_found = False
            for i in range(len(self.thresholds) - 1):
                if self.thresholds[i] <= value <= self.thresholds[i + 1]:
                    interval_counts.append((self.thresholds[i], self.thresholds[i + 1]))
                    interval_found = True
                    break
            if not interval_found:
                interval_counts.append(None)  # If value doesn't fall within any interval


        self.thresholds = self.thresholds
        self.intervals = []
        for k in sorted(interval_counts, key=lambda x: x[0]):
            self.intervals.append(k)

        return dict(Counter(interval_counts))

    def find_query_intervals(self, runtime):
        interval_found = False
        chosen_interval = None
        for i in range(len(self.thresholds) - 1):
            if self.thresholds[i] <= runtime <= self.thresholds[i + 1]:
                chosen_interval=(self.thresholds[i], self.thresholds[i + 1])
                interval_found = True
                break
        if not interval_found:
            return None  # If value doesn't fall within any interval
        return chosen_interval

    def print_intervals(self):
        counts = self.find_intervals()
        for k in sorted(counts, key = lambda x: x[0]):
            print(f"Inteval: {k}, #Q {counts[k]}")



class PercentileTimeIntervalExtractor(TimeIntervalExtractorBase):
    def __init__(self, run_times: list[float], percentiles: list[float]):
        """

        @param run_times: runtimes of queries in query log
        @param percentiles: list of percentiles to use for binning
        """
        self.percentiles = percentiles
        super().__init__(run_times)

    def create_interval_threshold(self) -> list[float]:
        return np.percentile(self.runtimes, self.percentiles).tolist()


def equi_width_bins(lst: list, num_bins):
    hist, bins = np.histogram(lst, bins=num_bins)
    print("Bin Edges:", bins)
    print("Histogram Counts:", hist)
    return bins


def binned_mean(lst, num_bins):
    result = scipy.stats.binned_statistic(lst, lst, bins=num_bins, statistic='mean')
    # 'sum for statistics another option
    bin_edges = result.bin_edges
    bin_means = result.statistic

    # Print the result
    print("Bin Edges:", bin_edges)
    print("Binned Mean:", bin_means)
    return bin_edges


def equalObs(x: list, nbin):
    """

    @type nbin: int
    """
    nlen = len(x)
    return np.interp(np.linspace(0, nlen, nbin + 1),
                     np.arange(nlen),
                     np.sort(x))


def equi_freq_bins(lst, num_bins):
    n, bins, patches = plt.hist(lst, equalObs(lst, num_bins), edgecolor='black')
    print("Bin Edges:", bins)
    print("Frequency:", n)

    return bins


def percentile_bins(lst, percentile_lst=None):  # [50,80,95] autowlm plot [50, 90, 99]
    if percentile_lst is None:
        percentile_lst = [50, 80, 95]
    return np.percentile(lst, percentile_lst)
