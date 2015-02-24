"""
Exploring the distribution of timestamps in a counting process with different
waiting time distributions.

@author Kevin Wilson - khwilson@gmail.com
"""
import random

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
    :type rv: scipy.stats.distributions.rv_frozen
    :param float T: The maximum timestamp to which to run the counting process
                    in a single draw
    :param int total_draws: The total number of draws of X described above to return
    :param int queue_size: The size of an internally managed queue of random numbers.
                          Not necessary to touch this; here for tuning speed.
    :return: A list of draws from X described above of size total_draws
    :rtype: list[float]
    """
    draws = []
    queue = rv.rvs(queue_size)
    queue_idx_start = 0
    while len(draws) < total_draws:
        draw, queue_idx_start = draw_process(rv, T, queue_size=queue_size, queue=queue,
                                             queue_idx_start=queue_idx_start)
        draws.append(random.choice(draw))
    return draws


def draw_process(rv, T, queue_size=100, queue=None, queue_idx_start=None):
    """
    Take a counting process defined by its waiting times, which are distributed according
    to a scipy rv_frozen variable, and run the process until time T. Return the list of
    event timestamps.

    :param rv: A scipy frozen distribution which represents the distribution
               of waiting times of the counting process
    :type rv: scipy.stats.distributions.rv_frozen
    :param float T: The maximum timestamp to which to run the counting process
    :param int queue_size: The size of an internally managed queue of random numbers.
                          Not necessary to touch this; here for tuning speed.
    :param numpy.ndarray queue: If not None, a queue of random draws from rv to use for the proces.
                                Not necessary to touch this; here for tuning speed.
    :param numpy.ndarray queue_idx_start: Required if queue is not None. The starting place in queue
                                          from which to grab random numbers.
    :return: A list of event timestamps from the process defined by rv
    :rtype: list[float]
    """
    timestamps = []
    queue_idx = queue_idx_start
    if not queue:
        queue = rv.rvs(queue_size)
        queue_idx = 0
    while True:
        if queue_idx == queue.size:
            queue_idx = 0
            queue[:] = rv.rvs(queue_size)
        val = queue[queue_idx]
        queue_idx += 1
        next_timestamp = (timestamps[-1] if timestamps else 0.0) + val
        if next_timestamp > T:
            break
        else:
            timestamps.append(next_timestamp)

    if queue_idx_start is not None:
        return timestamps, queue_idx
    else:
        return timestamps


def compute_increment(draw, a, b):
    """
    From a list of event timestamps in a counting process, count the number of events
    between a and b.

    :param list[float] draw: A list of timestamps drawn from a counting process
    :param float a: The left endpoint
    :param float b: The right endpoint
    :return: The increment between a and b
    :rtype: int
    """
    return sum(1 for d in draw if a < d < b)
