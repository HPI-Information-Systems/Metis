from contextlib import contextmanager
from dataclasses import dataclass
import time
import tracemalloc


@dataclass
class BenchmarkResults:
    runtime: float | None = None
    memory_end: float | None = None
    memory_peak: float | None = None


@contextmanager
def benchmark(runtime: bool, memory: bool, results: BenchmarkResults):
    """Context manager for benchmarking code execution time and memory usage."""
    start_time = time.perf_counter() if runtime else None
    tracemalloc.start() if memory else None
    try:
        yield
    finally:
        if start_time:
            results.runtime = time.perf_counter() - start_time
        if memory:
            results.memory_end, results.memory_peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
