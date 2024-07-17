import os
import tempfile
from typing import NamedTuple
import pytest
from ..dataset.beam_data_processor import (
    beam_data_processing_component,
    CsvInputData,
    CsvOutputData,
    BigQueryInputData,
    BigQuerySchemaField,
    BigQueryOutputData,
)


def test_csv_reader_writer():
    input_file = "kubeflow_components/tests/data/input.csv"
    processing_fn = "kubeflow_components.tests.conftest.processing_fn"
    init_fn = "kubeflow_components.tests.conftest.init_fn"
    # Output schema
    schema = NamedTuple(
        "OutputRow",
        [("A", int), ("B", int), ("C", int), ("D", int), ("E", int)],
    )
    with tempfile.TemporaryDirectory() as temp_dir:
        beam_data_processing_component(
            input_data=CsvInputData(file=input_file, batch_size=2),
            output_data=CsvOutputData(
                file=os.path.join(temp_dir, "output"), num_shards=3
            ),
            processing_fn=processing_fn,
            init_fn=init_fn,
        )


def test_bigquery_output_data():
    # Two modes of specifying BigQuery Output
    output_data1 = BigQueryOutputData("nfa-core-prod.public.videos", schema=[
        BigQuerySchemaField(name="id", type="STRING", mode="REQUIRED", description="video id"),
        BigQuerySchemaField(name="title", type="STRING", mode="NULLABLE")
    ])    
    output_data2 = BigQueryOutputData("nfa-core-prod.public.videos", schema=[
       {"name": "id", "type": "STRING", "mode": "REQUIRED", "description": "video id"},
       {"name":"title", "type":"STRING", "mode": "NULLABLE", "description": ""}
    ])
    
    assert output_data1.schema["fields"][0] == output_data2.schema["fields"][0]