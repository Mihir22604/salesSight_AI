"""
SalesSight AI Core Source Package.
Consolidated modules:
- data_pipeline: DataLoader, DataValidator, DataUnderstanding, DataCleaner, SalesAggregator
- eda: EDAAnalyzer, EDAVisualizer
- utils: get_logger, set_seed, SchemaValidationError, DataIntegrityError
"""
# Ensure OpenMP/BLAS runtime is initialized cleanly on Windows
try:
    import lightgbm
except ImportError:
    pass

from .utils import get_logger, set_seed, SchemaValidationError, DataIntegrityError
from .data_pipeline import (
    DataLoader,
    DataValidator,
    DataUnderstanding,
    DataCleaner,
    SalesAggregator,
    run_data_pipeline,
)
from .eda import EDAAnalyzer, EDAVisualizer, run_eda_pipeline
from .correlation import CorrelationAnalyzer, CorrelationVisualizer, run_correlation_pipeline
from .features import FeatureBuilder, FeatureTransformer, LeakageGuard, run_feature_pipeline
from .feature_selection import FeatureSelector, FeatureSelectionEvaluator, run_feature_selection_pipeline
from .split import TimeSeriesSplitter, SplitVisualizer, run_split_pipeline
from .models import (
    ModelFactory,
    ModelEvaluator,
    ModelVisualizer,
    DemandDiscretizer,
    SeasonalNaiveBaseline,
    AlwaysZeroBaseline,
    TrainMeanBaseline,
    SARIMAForecastModel,
    run_models_pipeline,
)
from .tuning import (
    BenchmarkEvaluator,
    ObjectiveComparator,
    HyperparameterTuner,
    GranularityEvaluator,
    WinnerDecisionEngine,
    TuningVisualizer,
    run_tuning_pipeline,
)
from .selection import (
    TunedModelEvaluator,
    TunedClassificationVisualizer,
    MultiCriteriaSelectionEngine,
    run_stage11_selection_pipeline,
)
from .persistence import (
    ProvenanceAuditEngine,
    ResidualCalibrator,
    ProductionModelPackage,
    ModelPersistenceManager,
    run_stage12_persistence_pipeline,
)

__all__ = [
    "get_logger",
    "set_seed",
    "SchemaValidationError",
    "DataIntegrityError",
    "DataLoader",
    "DataValidator",
    "DataUnderstanding",
    "DataCleaner",
    "SalesAggregator",
    "run_data_pipeline",
    "EDAAnalyzer",
    "EDAVisualizer",
    "run_eda_pipeline",
    "CorrelationAnalyzer",
    "CorrelationVisualizer",
    "run_correlation_pipeline",
    "FeatureBuilder",
    "FeatureTransformer",
    "LeakageGuard",
    "run_feature_pipeline",
    "FeatureSelector",
    "FeatureSelectionEvaluator",
    "run_feature_selection_pipeline",
    "TimeSeriesSplitter",
    "SplitVisualizer",
    "run_split_pipeline",
    "ModelFactory",
    "ModelEvaluator",
    "ModelVisualizer",
    "DemandDiscretizer",
    "SeasonalNaiveBaseline",
    "AlwaysZeroBaseline",
    "TrainMeanBaseline",
    "SARIMAForecastModel",
    "run_models_pipeline",
    "BenchmarkEvaluator",
    "ObjectiveComparator",
    "HyperparameterTuner",
    "GranularityEvaluator",
    "WinnerDecisionEngine",
    "TuningVisualizer",
    "run_tuning_pipeline",
    "TunedModelEvaluator",
    "TunedClassificationVisualizer",
    "MultiCriteriaSelectionEngine",
    "run_stage11_selection_pipeline",
]
