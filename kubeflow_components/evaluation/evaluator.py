import json
from dataclasses import dataclass
from typing import List, Dict, Optional, Union, Callable, Type, Iterable, Literal
import apache_beam as beam
from kubeflow_components.dataset.beam_data_processor.beam_data_processor import (
    BaseInputData,
)
from kubeflow_components.evaluation.metrics import BaseMetric


# %% TFMA-like Model Eval without Protobuf
# and use models in addition to Tensorflow


@dataclass
class ModelSpec:
    name: str
    inference_fn: Union[str, Callable]
    init_fn: Union[str, Callable]


@dataclass
class MetricThreshold:
    # Direction Enum
    HIGHER_IS_BETTER = "higher_is_better"
    LOWER_IS_BETTER = "lower_is_better"

    # Inner key of the metric, if the metric
    # returns a dictionary rather than a single value
    # e.g. top_k metrics.
    metric_key: str = None
    # Slice: apply threshold to a metric computed on 
    # a specific slice of the input data
    slice_key: str = None
    # Metric needs to be >= this value
    lower: Union[float, int] = None
    # Metric needs to be <= this value
    upper: Union[float, int] = None
    # Minimum examples needed for a valid metric
    min_examples: int = None
    # Direction, whether higher or lower is better
    direction: Literal["higher_is_better", "lower_is_better"] = "higher_is_better"


@dataclass
class MetricSpec:
    name: str
    # path ot the metric class
    module_path: Union[str, Type[BaseMetric]]
    # Determines if the evaluator should give blessing
    threshold: Optional[MetricThreshold] = None

@dataclass
class SliceConfig:
    pass


@dataclass
class DataSpec:
    input_data: BaseInputData
    slices: Optional[List[SliceConfig]] = None


class EvalConfig:
    def __init__(
        self,
        model: ModelSpec,
        metrics: List[MetricSpec],
        data: DataSpec,
        baseline_model: Optional[ModelSpec] = None,
    ):
        """
        Evaluation Configuration.

        Args:
            model (ModelSpec): model configuration
            metrics (List[MetricSpec]): List of metrics to compute
            data (DataSpec): evaluation data configuration
            baseline_model (Optional[ModelSpec], optional): baseline
                model to compare to. This is useful to calculate
                metrics that compares baseline models, such as
                the difference in the set of predictions. Defaults to
                None.
        """
        if data.slices is None:
            data.slices = [SliceConfig()]
        
        self.model = model
        self.metrics = metrics
        self.data = data
        self.baseline_model = baseline_model


# %% Metric writer
class MetricWriter(beam.PTransform):
    def __init__(self, output_file):
        self.output_file = output_file

    def expand(self, pcoll):
        return (
            pcoll
            | beam.Map(lambda x: json.dumps(x))
            | beam.io.WriteToText(self.output_file)
        )


# %% Model prediction
class ModelInference(beam.PTransform):
    """Batch job to make inference on model."""

    def __init__(self):
        pass

    def expand(self, pcoll):
        return pcoll
