import json
from typing import List, Dict, Tuple, Optional
import apache_beam as beam

from kubeflow_components.dataset.beam_data_processor import (
    ReadBigQueryData,
    ReadTFRecordData,
)

# Read Data
# Make Inference
# %% Write metric file


class MetricWriter(beam.PTransform):
    def __init__(self, output_file):
        self.output_file = output_file

    def expand(self, pcoll):
        return (
            pcoll
            | beam.Map(lambda x: json.dumps(x))
            | beam.io.WriteToText(self.output_file)
        )
