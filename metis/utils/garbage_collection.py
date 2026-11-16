import gc
from contextlib import contextmanager
from typing import Iterator


@contextmanager
def gc_paused() -> Iterator[None]:
    """
    Pause Python's cyclic garbage collector for the duration of the block.

    Metrics that emit one result per row or cell allocate millions of objects,
    and each allocation burst triggers a collection pass that rescans every
    live result, which slows them down by up to ~2x. Reference counting still
    frees objects while paused; only cycle detection is deferred until the
    block exits. The collector is restored only if it was enabled on entry.

    :return: An iterator yielding nothing, for use as a context manager.
    """
    was_enabled = gc.isenabled()
    gc.disable()
    try:
        yield
    finally:
        if was_enabled:
            gc.enable()
