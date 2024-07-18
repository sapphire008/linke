"""Make sure to test from root directory"""

import os
import tempfile
import shutil
from typing import NamedTuple, List
import pytest
import pandas as pd
from ..runner.local_runner import LocalPipelineRunner
from ..dataset.beam_data_processor import (
    beam_data_processing_fn,
    CsvInputData,
    CsvOutputData,
    BigQueryInputData,
    BigQuerySchemaField,
    BigQueryOutputData,
)
from ..dataset.beam_data_processor_component import (
    beam_data_processing_component,
)


@pytest.mark.skip(
    reason="working already, skip for now during development"
)
def test_csv_reader_writer():
    input_file = "kubeflow_components/tests/data/input.csv"
    processing_fn = (
        "kubeflow_components.tests.conftest.csv_processing_fn"
    )
    init_fn = "kubeflow_components.tests.conftest.csv_init_fn"

    with tempfile.TemporaryDirectory() as temp_dir:
        output_file = os.path.join(temp_dir, "output")
        beam_data_processing_fn(
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


@pytest.mark.skip(
    reason="working already, skip for now during development"
)
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


@pytest.mark.skip(reason="No longer have access to the test tables. Need to reconfigure in the future.")
def test_bigquery_reader_writer():
    query = """
    SELECT * EXCEPT(row_id) 
    FROM (
        SELECT id, title, tags, 
            ROW_NUMBER() OVER(PARTITION BY id) AS row_id
        FROM `nfa-core-prod.public.videos`
        WHERE age_rating = "mpa:pg-13"
    )
    WHERE row_id = 1
    """
    # Make sure this schema is in the same order as the output from the processing_fn
    schema = [
        BigQuerySchemaField(name="id", type="STRING", mode="REQUIRED"),
        BigQuerySchemaField(
            name="transformed_title", type="STRING", mode="NULLABLE"
        ),
        BigQuerySchemaField(
            name="num_tags", type="INT64", mode="NULLABLE"
        ),
    ]

    beam_data_processing_fn(
        input_data=BigQueryInputData(sql=query, batch_size=3),
        output_data=BigQueryOutputData(
            output_table="nfa-core-prod.temp_dataset.test_beam_writer",
            schema=schema,
        ),
        processing_fn="kubeflow_components.tests.conftest.bq_processing_fn",
        init_fn="kubeflow_components.tests.conftest.csv_init_fn",
        beam_pipeline_args=[
            "--runner=DirectRunner",
            "--temp_location=gs://ml-pipeline-runs/bigquery",
            "--project=nfa-core-prod",
        ],
    )


@pytest.mark.skip(
    reason="working already, skip for now during development"
)
def test_beam_data_processing_single_component():
    """No input/output artifacts. Just static files linked"""
    input_file = "kubeflow_components/tests/data/input.csv"
    # Initialize the runner
    runner = LocalPipelineRunner(runner="subprocess")
    with tempfile.TemporaryDirectory() as temp_dir:
        # output_file = os.path.join(temp_dir, "output")
        # make payload
        payload = {
            "processing_fn": "kubeflow_components.tests.conftest.processing_fn",
            "init_fn": "kubeflow_components.tests.conftest.init_fn",
            "input_data": CsvInputData(
                file=input_file, batch_size=2
            ).as_dict(),
            "output_data": CsvOutputData(
                # file=output_file,
                num_shards=3,
                headers=["A", "B"],
            ).as_dict(),
        }
        # Run the pipeline
        task = runner.create_run(
            beam_data_processing_component, payload
        )

        # Check that there are 3 files
        files = os.listdir(task.output.uri)
        assert len(files) == 3, "Expected 3 files"
        shutil.rmtree(task.output.uri)  # clean up
