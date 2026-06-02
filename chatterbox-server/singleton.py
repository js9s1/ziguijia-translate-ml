"""Thread-safe singleton decorator.

Usage::

    from singleton import singleton

    @singleton
    class MyService:
        def __init__(self):
            ...

    # Access instance:
    svc = MyService()

    # Reset for fork (e.g., gunicorn post_fork):
    MyService.clear()
"""

import threading


def singleton(cls):
    """Decorator: make *cls* a thread-safe singleton.

    The first call creates the instance; subsequent calls return the
    same instance.  Constructor arguments are forwarded to ``__init__``
    on first creation only.

    Call ``.clear()`` on the wrapper to reset the instance
    (e.g., after ``os.fork``).
    """
    _lock = threading.Lock()
    _instance = [None]

    def _get_instance(*args, **kwargs):
        if _instance[0] is None:
            with _lock:
                if _instance[0] is None:
                    _instance[0] = cls(*args, **kwargs)
        return _instance[0]

    def _clear():
        with _lock:
            _instance[0] = None

    _get_instance.clear = _clear
    _get_instance.__name__ = cls.__name__
    _get_instance.__qualname__ = cls.__qualname__
    _get_instance.__doc__ = cls.__doc__
    return _get_instance
