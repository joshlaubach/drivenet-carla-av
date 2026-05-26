"""Project-level pytest configuration.

Sets the Agg matplotlib backend before any test module is imported, so that
BenchmarkReport tests work in headless CI environments (no Tk/display needed).
"""
import matplotlib
matplotlib.use("Agg")
