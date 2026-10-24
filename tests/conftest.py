"""
Pytest global configuration for SalesSight AI test suite.
Ensures headless matplotlib (Agg backend) and clean OpenMP initialization across all test runs.
"""
import matplotlib
matplotlib.use("Agg")

try:
    import lightgbm
except ImportError:
    pass
