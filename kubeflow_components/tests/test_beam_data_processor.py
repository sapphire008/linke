import os
import tempfile
from typing import NamedTuple
import pytest
import pandas as pd
from ..dataset.beam_data_processor import (
    beam_data_processing_component,
    CsvInputData,
    CsvOutputData,
    BigQueryInputData,
    BigQuerySchemaField,
    BigQueryOutputData,
)

from pdb import set_trace

@pytest.mark.skip(reason="working already, skip for now during development")
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
        output_file = os.path.join(temp_dir, "output")
        beam_data_processing_component(
            input_data=CsvInputData(file=input_file, batch_size=2),
            output_data=CsvOutputData(
                file=output_file,
                num_shards=3,
                headers=["A", "B"],
            ),
            processing_fn=processing_fn,
            init_fn=init_fn,
        )
        # Check results
        df = []
        for ii, file in enumerate(os.listdir(temp_dir)):
            if not file.endswith(".csv"):
                continue
            df.append(pd.read_csv(os.path.join(temp_dir, file)))
        df = pd.concat(df)
        assert ii == 2, "Expecting 3 shards"
        assert (
            df.columns[0] == "A" and df.columns[1] == "B"
        ), f"Expecting only 2 columns, but got {df.columns}"
        df_input = pd.read_csv(input_file)
        assert (
            df_input.shape[0] == df.shape[0]
        ), "Expecting same number of outputs as inputs"


@pytest.mark.skip(reason="working already, skip for now during development")
def test_bigquery_output_data():
    # Two modes of specifying BigQuery Output
    output_data1 = BigQueryOutputData(
        "nfa-core-prod.public.videos",
        schema=[
            BigQuerySchemaField(
                name="id",
                type="STRING",
                mode="REQUIRED",
                description="video id",
            ),
            BigQuerySchemaField(
                name="title", type="STRING", mode="NULLABLE"
            ),
        ],
    )
    output_data2 = BigQueryOutputData(
        "nfa-core-prod.public.videos",
        schema=[
            {
                "name": "id",
                "type": "STRING",
                "mode": "REQUIRED",
                "description": "video id",
            },
            {
                "name": "title",
                "type": "STRING",
                "mode": "NULLABLE",
                "description": "",
            },
        ],
    )

    assert (
        output_data1.schema["fields"][0]
        == output_data2.schema["fields"][0]
    )


# @pytest.mark.skip(reason="working already, skip for now during development")
def test_bigquery_reader_writer():
    pass
