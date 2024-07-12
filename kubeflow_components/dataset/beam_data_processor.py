"""
Apache Beam based data procesor. The overall architecture is:

Reader -> Custom processor class -> Writer

Reader and Writer are a set of supported components by Apache Beam
"""

from typing import Literal, NamedTuple, List, Dict, Union, Generator
from collections import namedtuple
import pandas as pd
import apache_beam as beam
from apache_beam.options.pipeline_options import PipelineOptions
from apache_beam.io.textio import ReadFromCsv, WriteToCsv, WriteToText
from apache_beam.transforms import util as beam_util
from pdb import set_trace


def processing_fn(inputs, model=None, config={}):
    pass


class DataProcessingDoFn(beam.DoFn):
    def __init__(self, config={}):
        pass

    def process(
        self, element
    ) -> Generator[Union[pd.DataFrame, pd.Series, List[Dict], Dict], None, None]:
        yield element


class ReadCSVData(beam.PTransform):
    """
    Read CSV and convert into a dict/df instead of PCollection

    * file_pattern: A single csv file or a file pattern (glob pattern)
    * format: output format, either as a Pandas dataframe or as a dict
    * min_batch_size: minimum batch size. If not None, the output will
        be a batch of records rather than a single record.
    * max_batch_size: maximum batch size if using batch. Default to 1024.
    * **kwargs: addition keyword arguments used to read csv. See
        https://beam.apache.org/releases/pydoc/current/apache_beam.io.textio.html?highlight=readfromcsv#apache_beam.io.textio.ReadFromCsv
    """

    def __init__(
        self,
        file_pattern: str,
        format: Literal["dataframe", "dict"] = "dict",
        min_batch_size: int = None,
        max_batch_size: int = 1024,
        **kwargs,  # for beam.ReadFromCsv
    ):
        super().__init__()
        self.file_pattern = file_pattern
        self.format = format
        self.min_batch_size = min_batch_size
        self.max_batch_size = max_batch_size
        self.kwargs = kwargs

    def _convert(self, x):
        import pandas as pd

        if self.format == "dataframe":
            if self.min_batch_size is None:
                return pd.DataFrame([x]).loc[0]  # pd.Series
            else:
                return pd.DataFrame(x)
        elif self.format == "dict":
            if self.min_batch_size is None:
                return pd.DataFrame([x]).to_dict("records")[0]
            else:
                return pd.DataFrame(x).to_dict("records")
        else:
            raise (ValueError(f"Unrecognized format {self.format}"))

    def expand(self, pcoll):
        pcoll = pcoll | "Read CSV" >> ReadFromCsv(
            self.file_pattern, **self.kwargs
        )
        # Batching when needed
        if self.min_batch_size is not None:
            pcoll = pcoll | "Batching" >> beam.BatchElements(
                min_batch_size=self.min_batch_size,
                max_batch_size=self.max_batch_size,
            )

        return pcoll | "Convert to Dict" >> beam.Map(self._convert)


class WriteCsvData(beam.PTransform):
    def __init__(
        self, file_pattern: str, schema: NamedTuple = None, **kwargs
    ):
        self.file_pattern = file_pattern
        self.schema = schema
        self.kwargs = kwargs  # for write to text
        self.header = None
        if schema is not None:
            self.header = ",".join(schema._fields)

    def to_csv_string(self, element):
        import io
        import pandas as pd

        if isinstance(element, list):
            element = pd.DataFrame(element)
        elif isinstance(element, dict):
            element = pd.DataFrame([element])
        elif isinstance(element, pd.DataFrame):
            pass
        elif isinstance(element, pd.Series):
            element = element.to_frame().T
        else:
            raise (
                TypeError(
                    f"Unrecognized input data type {type(element)}"
                )
            )

        # Convert to csv string
        output = io.StringIO()
        pd.DataFrame(element).to_csv(output, header=None, index=False)
        return output.getvalue().strip()

    def expand(self, pcoll):
        return (
            pcoll
            | "Convert Dict to PColl" >> beam.Map(self.to_csv_string)
            | "Write to csv"
            >> WriteToText(
                self.file_pattern,
                file_name_suffix=".csv",
                header=self.header,
                **self.kwargs,
            )
        )


def run(beam_pipeline_args=["--runner=DirectRunner"]):
    options = PipelineOptions(flags=beam_pipeline_args)
    input_file = "/Users/edward/Documents/Scripts/kubeflow-components/kubeflow_components/tests/data/input.csv"
    output_file = "/Users/edward/Documents/Scripts/kubeflow-components/kubeflow_components/tests/data/output"

    # Output schema
    schema = NamedTuple(
        "OutputRow",
        [("A", int), ("B", int), ("C", int), ("D", int), ("E", int)],
    )

    with beam.Pipeline(options=options) as p:
        _ = (
            p
            | "Read CSV"
            >> ReadCSVData(input_file, format="dict", min_batch_size=2)
            | "Process Data" >> beam.ParDo(DataProcessingDoFn())
            | "Write Data"
            >> WriteCsvData(output_file, num_shards=3, schema=schema)
        )


# %% Kubeflow component

if __name__ == "__main__":
    run()
