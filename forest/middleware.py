from functools import wraps


def middleware(func):
    """Decorator to curry middleware functions

    .. note:: depending on context *args can be
              (store,) or (self, store)
    """
    @wraps(func)
    def x(*args):
        def y(next_method):
            def z(action):
                func(*args, next_method, action)
            return z
        return y
    return x
