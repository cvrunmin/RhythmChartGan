import numpy as np

def linear_ease(start, end, t):
    d = end - start
    return start + d * t

def inverse_lerp(start, end, t):
    return (t - start) / (end - start)


def cubic_ease(start, end, ctrl_1, ctrl_2, t):
    # from the expanded equation from Wikipedia
    return (1-t) ** 3 * start + 3 * (1-t)**2 * t * ctrl_1 + 3 * (1-t) * t ** 2 * ctrl_2 + t ** 3 * end
