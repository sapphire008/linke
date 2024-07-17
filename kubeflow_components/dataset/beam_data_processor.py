"""
Apache Beam based data procesor. The overall architecture is:

Reader -> Custom processor class -> Writer

Reader and Writer are a set of supported components by Apache Beam
"""

import sys
from dataclasses import dataclass, field
from typing import (
    Literal,
    NamedTuple,
    List,
    Dict,
    Union,
    Generator,
    Callable,
    Type,
)
import keyword
import importlib
import pandas as pd
import apache_beam as beam
from apache_beam.options.pipeline_options import PipelineOptions
from apache_beam.io.filesystem import CompressionTypes
from apache_beam.io.textio import ReadFromCsv, WriteToText
from apache_beam.io.gcp.bigquery import (
    ReadFromBigQuery,
    WriteToBigQuery,
    BigQueryDisposition,
)
from kfp import dsl

from pdb import set_trace


# %% Custom DoFn
class WeakRef:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            assert self._check_valid_attribute_name(k), (
                f"{k} is not a valid config key from initialization. "
                "Make sure to use keys that can be used "
                "for object attributes."
            )
            setattr(self, k, v)

    def _check_valid_attribute_name(self, name: str):
        return name.isidentifier() and not keyword.iskeyword(name)


class DataProcessingDoFn(beam.DoFn):
    def __init__(
        self,
        processing_fn: Union[
            str, Callable[[List[Dict], Dict], List[Dict]]
        ],
        init_fn: Union[str, Callable[[], Dict]],
        config: Dict = {},
    ):
        self._shared_handle = beam.utils.shared.Shared()
        self.processing_fn = (
            self.import_function(processing_fn)
            if isinstance(processing_fn, str)
            else processing_fn
        )
        self.init_fn = (
            self.import_function(init_fn)
            if isinstance(init_fn, str)
            else init_fn
        )
        assert (
            "weak_ref" not in config
        ), "weak_ref is a reserved field name for `config`"
        self.config = config

    @staticmethod
    def import_function(path: str):
        try:
            module_path, function_name = path.rsplit(".", 1)
            module = importlib.import_module(module_path)
            return getattr(module, function_name)
        except (ImportError, AttributeError) as e:
            raise ImportError(
                f"Error importing function {path}: {str(e)}"
            )

    def setup(self):
        def _initialize() -> WeakRef:
            # Call initialization function
            result: Dict = self.init_fn()
            # Convert to WeakRef
            return WeakRef(**result)

        # Create reference objects
        self.config["weak_ref"] = self._shared_handle.acquire(
            _initialize
        )

    def process(
        self, element
    ) -> Generator[
        Union[pd.DataFrame, pd.Series, List[Dict], Dict], None, None
    ]:

        # Call the processing function
        outputs = self.processing_fn(element, config=self.config)

        yield outputs


# %%
@dataclass(kw_only=True)
class BaseInputData:
    batch_size: int = field(default=None)
    format: Literal["dict", "dataframe"] = field(default="dict")



@dataclass(kw_only=True)
class BaseOutputData:
    pass


# %% CSV Reader and Writers
@dataclass(kw_only=True)
class CsvInputData(BaseInputData):
    file: str
    columns: List[str] = None  # list of columns to read


@dataclass(kw_only=True)
class CsvOutputData(BaseOutputData):
    file: str
    num_shards: int = 1  # csv shards
    headers: List[str] = None  # list of output headers
    compression_type: CompressionTypes = CompressionTypes.UNCOMPRESSED


class ReadCsvData(beam.PTransform):
    """
    Read CSV and convert into a dict/df instead of PCollection

    * file_pattern: A single csv file or a file pattern (glob pattern)
    * format: output format, either as a Pandas dataframe or as a dict
    * min_batch_size: minimum batch size. If not None, the output will
        be a batch of records rather than a single record.
    * max_batch_size: maximum batch size if using batch. Default to 1024.
    * columns: list of columns to read only
    * **kwargs: addition keyword arguments used to read csv. See
        https://beam.apache.org/releases/pydoc/current/apache_beam.io.textio.html?highlight=readfromcsv#apache_beam.io.textio.ReadFromCsv
    """

    def __init__(
        self,
        file_pattern: str,
        format: Literal["dataframe", "dict"] = "dict",
        min_batch_size: int = None,
        max_batch_size: int = 1024,
        columns: List[str] = None,
        **kwargs,  # for beam.ReadFromCsv
    ):
        super().__init__()
        self.file_pattern = file_pattern
        self.format = format
        self.min_batch_size = min_batch_size
        self.max_batch_size = max_batch_size
        self.columns = columns
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
            self.file_pattern, usecols=self.columns, **self.kwargs
        )
        # Batching when needed
        if self.min_batch_size is not None:
            pcoll = pcoll | "Batching" >> beam.BatchElements(
                min_batch_size=self.min_batch_size,
                max_batch_size=self.max_batch_size,
            )

        return pcoll | "Convert Format" >> beam.Map(self._convert)


class WriteCsvData(beam.PTransform):
    def __init__(
        self,
        file_pattern: str,
        headers: List[str] = None,
        compression_type=CompressionTypes.UNCOMPRESSED,
        **kwargs,
    ):
        super().__init__()
        self.file_pattern = file_pattern
        self.compression_type = compression_type
        self.kwargs = kwargs  # for write to text
        self.headers = ",".join(headers) if headers else None

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
                header=self.headers,
                compression_type=self.compression_type,
                **self.kwargs,
            )
        )


# %% BigQuery
@dataclass
class BigQueryInputData(BaseInputData):
    sql: str  # Input sql string
    temp_dataset: str  # project_id.temp_dataset


@dataclass
class BigQueryOutputData(BaseOutputData):
    output_table: str  # project_id.dataset.table
    mode: BigQueryDisposition = (
        BigQueryDisposition.WRITE_APPEND
    )  # BigQuery write mode
    _schema: Type[NamedTuple] = None

    @staticmethod
    def namedtuple_to_bq_schema(schema):
        schema = NamedTuple(
            "OutputRow",
            [
                ("A", int),
                ("B", int),
                ("C", int),
                ("D", int),
                ("E", int),
            ],
        )


class ReadBigQueryData(beam.PTransform):
    def __init__(
        self,
        query: str,
        format: Literal["dataframe", "dict"] = "dict",
        min_batch_size: int = None,
        max_batch_size: int = 1024,
        use_standard_sql: bool = True,
        **kwargs,
    ):
        super().__init__()
        self.query = query
        self.format = format
        self.min_batch_size = min_batch_size
        self.max_batch_size = max_batch_size
        self.use_standard_sql = use_standard_sql
        self.kwargs = kwargs

    def _convert_to_df(self, x: Union[Dict, List[Dict]]):
        import pandas as pd

        if self.min_batch_size is None:
            return pd.Series(x)  # pd.Series
        else:
            return pd.DataFrame(x)

    def expand(self, pcoll):
        pcoll = pcoll | "Read BigQuery" >> ReadFromBigQuery(
            query=self.query,
            use_standard_sql=self.use_standard_sql,
            **self.kwargs,
        )
        # Batching when needed
        if self.min_batch_size is not None:
            pcoll = pcoll | "Batching" >> beam.BatchElements(
                min_batch_size=self.min_batch_size,
                max_batch_size=self.max_batch_size,
            )
        # Default outputs dictionary
        if self.format == "dataframe":
            return pcoll | "Convert Format" >> beam.Map(
                self._convert_to_df
            )
        else:
            return pcoll


class WriteBigQueryData(beam.PTransform):
    def __init__(
        self,
        output_table: str,
        schema: Dict,
        write_disposition: BigQueryDisposition = BigQueryDisposition.WRITE_APPEND,
        **kwargs,
    ):
        self.table = self.normalize_table_reference(output_table)
        self.schema = schema  # bigquery specific schema
        self.write_disposition = write_disposition
        self.kwargs = kwargs

    @staticmethod
    def normalize_table_reference(output_table):
        parts = output_table.replace(":", ".").split(".")
        if len(parts) != 3:
            raise ValueError(
                "Invalid input format. Expected 'project_id:dataset.table' "
                "or 'project_id.dataset.table'"
            )
        return f"{parts[0]}:{parts[1]}.{parts[2]}"

    def expand(self, pcoll):
        return pcoll | "Write to BigQuery" >> WriteToBigQuery(
            table=self.table,
            schema=self.schema,
            write_disposition=self.write_disposition,
            **self.kwargs,
        )


# %% Create the kubeflow component
def beam_data_processing_component(
    input_data: BaseInputData,
    output_data: BaseOutputData,
    processing_fn: str,
    init_fn: str,
    beam_pipeline_args: List[str] = ["--runner=DirectRunner"],
) -> None:
    options = PipelineOptions(flags=beam_pipeline_args)

    # Create beam pipeline
    with beam.Pipeline(options=options) as p:
        # inputs
        if isinstance(input_data, CsvInputData):
            pcoll = p | "Read CSV" >> ReadCsvData(
                input_data.file,
                format=input_data.format,
                min_batch_size=input_data.batch_size,
            )
        elif isinstance(input_data, BigQueryInputData):
            pcoll = p | "Read BigQuery" >> ReadBigQueryData(
                format=input_data.format,
                min_batch_size=input_data.batch_size,
                temp_dataset=input_data.temp_dataset,
            )

        # Run the
        pcoll = pcoll | "Process Data" >> beam.ParDo(
            DataProcessingDoFn(
                processing_fn=processing_fn,
                init_fn=init_fn,
            )
        )

        # Output
        if isinstance(output_data, CsvOutputData):
            pcoll = pcoll | "Write CSV Data" >> WriteCsvData(
                output_data.file,
                num_shards=output_data.num_shards,
                headers=output_data.headers,
            )
        elif isinstance(output_data, BigQueryOutputData):
            pcoll = pcoll | "Write BigQuery Data" >> WriteBigQueryData(
                output_table=output_data.output_table,
                schema=output_data.schema,
                write_disposition=output_data.mode,
            )
