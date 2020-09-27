import numpy as np

# way faster than np.random.choice
# arr is an array of probabilities, should sum to 1
def sample(arr):
    r = np.random.rand()
    s = 0
    for i, p in enumerate(arr):
        s += p
        if s > r or s == 1:
            return i

    # worst case if we run into floating point error, just return the last element
    # we should never get here
    return len(arr) - 1

# faster than np.random.choice
def choice(arr, size = None):
    ind = np.random.randint(0, len(arr), size=size)
    return np.array(arr)[ind]

def argmax(arr):
    ties = []
    top = -np.inf
    for i, a in enumerate(arr):
        if a > top:
            top = a
            ties = [i]
        elif a == top:
            ties.append(i)

    return choice(ties)