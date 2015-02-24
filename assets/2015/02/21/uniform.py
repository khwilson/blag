"""
Exploring the distribution of timestamps in a counting process with different
waiting time distributions.

@author Kevin Wilson - khwilson@gmail.com
"""
import random

import numpy as np
from scipy import stats as st


def draw_timestamps(rv, T, total_draws, queue_size=10000):
    """
    Draw total_draws from the random variable X described below:
    * Let CP be the counting process with stationary and independent
      increments whose events do not happen simultaneously and whose waiting
      times are distributed according to rv
    * Let P be the set-valued random variable whose realizations are the event
      timestamps of a draw from CP which occur before time T
    * Let X be the real-valued random variable whose realizations are an element
      of P chose uniformly at random, assuming |P| > 0. If P is the empty set,
      then let X be -1.

    :param rv: A scipy frozen distribution which represents the distribution
               of waiting times of the counting process
    :type rv: st.distributions.rv_frozen
    :param float T: The maximum timestamp to which to run the counting process
                    in a single draw
    :param int total_draws: The total number of draws of X described above to return
    :param in queue_size: The size of an internally managed queue of random numbers.
                          Not necessary to touch this; here for tuning speed.
    :return: A list of draws from X described above of size total_draws
    :rtype: list[float]
    """
    draws = []

    timestamps = []
    queue = rv.rvs(queue_size)
    queue_idx = 0
    while len(draws) < total_draws:
        if queue_idx == queue.size:
            queue_idx = 0
            queue[:] = rv.rvs(queue_size)
        val = queue[queue_idx]
        queue_idx += 1
        next_timestamp = (timestamps[-1] if timestamps else 0.0) + val
        if next_timestamp > T:
            if timestamps:
                draws.append(random.choice(timestamps))
            else:
                draws.append(-1)
            timestamps = []
        else:
            timestamps.append(next_timestamp)

    return draws
