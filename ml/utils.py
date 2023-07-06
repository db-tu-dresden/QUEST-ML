from contextlib import contextmanager


@contextmanager
def optional(condition, context_manager, *args, **kwargs):
    if condition:
        with context_manager(*args, **kwargs):
            yield
    else:
        yield
