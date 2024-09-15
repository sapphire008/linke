import os
import tempfile
import numpy as np
import pandas as pd
from linke.dataset.beam_data_processor.beam_data_processor import (
    CsvInputData,
)
from linke.evaluation.evaluator import (
    create_evaluation_pipeline,
    EvalConfig,
    DataSpec,
    SliceConfig,
    ModelSpec,
    MetricSpec,
    MetricThreshold,
)

from pdb import set_trace
import warnings
warnings.filterwarnings("ignore")

def label_transform_fn(labels, config={}):
    alphabet = "abcdefghij"
    transformed_labels = []
    for val in labels:
        label = [alphabet[ii] for ii in range(val)]
        np.random.shuffle(label)
        transformed_labels.append(label)
    return transformed_labels

def inference_fn(inputs, config={}):
    alphabet = "abcdefghij"
    predictions = []        
    # Faking predictions
    for _ in range(len(inputs["A"])):
        prediction =list(alphabet)
        np.random.shuffle(prediction)
        predictions.append(prediction)
    predictions = np.array(predictions)
    return predictions

def setup_fn():
    return {}


class TestEvaluator:
    def setup_method(self, method=None):
        self.data_spec = DataSpec(
            input_data=CsvInputData(
                # Not really doing anything
                file="linke/tests/data/input.csv", batch_size=2
            ),
            label_key="E",
            slices = None,
        )
        self.model_spec = ModelSpec(
            name="model_1",
            inference_fn=inference_fn,
            setup_fn=setup_fn,
            label_transform_fn=label_transform_fn,
        )
        # Create metric from module_path
        self.metric_hit_ratio = MetricSpec(
            name="hit_ratio",
            metric="linke.evaluation.metrics.HitRatioTopK",
            config={"top_k": [1, 4, 5]},
        )
        self.metric_ndcg = MetricSpec(
            name="ndcg",
            metric="linke.evaluation.metrics.NDCGTopK",
            config={"top_k": [1, 4, 5]},
        )
    
    def teardown_method(self, method=None):
        # remove temp data file
        pass

    def test_evaluation_pipeline(self):
        """Test evaluation pipeline from end-to-end."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_dir = "./"
            create_evaluation_pipeline(
                eval_config=EvalConfig(
                    model=self.model_spec,
                    metrics=[self.metric_hit_ratio, self.metric_ndcg],
                    data=self.data_spec,
                ),
                metric_result=os.path.join(temp_dir, "metric_result.json"),
                blessing_result=os.path.join(temp_dir, "blessing_result.json"),
                beam_pipeline_args=["--runner=DirectRunner"]
            )
