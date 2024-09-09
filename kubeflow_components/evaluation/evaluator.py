import json
from dataclasses import dataclass
from typing import List, Optional
import apache_beam as beam


# %% TFMA-like Model Eval without Protobuf
# and use models in addition to Tensorflow

@dataclass
class ModelSpec:
    pass

@dataclass
class MetricSpec:
    pass


class MetricConfig:
    def __init__(self):
        pass

class SliceConfig:
    def __init__(self):
        pass

@dataclass
class DataSpec:
    input_data: List
    slices: List[SliceConfig] = [SliceConfig()]


class EvalConfig:
    def __init__(
        self,
        model: ModelSpec,
        metrics: List[MetricSpec],
        data: DataSpec,
        slices: List[SliceConfig] = [SliceConfig()],
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
        self.model = model
        self.data = data
        self.baseline_model = baseline_model
        self.metrics = metrics
        self.slices = slices


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
