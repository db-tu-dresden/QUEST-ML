from contextlib import contextmanager


@contextmanager
def optional(condition, context_manager):
    if condition:
        with context_manager():
            yield
    else:
        yield
