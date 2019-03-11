import time


def timed(func):
    """decorator used to time functions"""
    def wrapped(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        duration = end - start
        print('{} ran for {} seconds'.format(str(func), duration))
        return result
    return wrapped
