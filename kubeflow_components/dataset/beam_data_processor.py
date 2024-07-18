"""
Apache Beam based data procesor. The overall architecture is:

Reader -> Custom processor class -> Writer

Reader and Writer are a set of supported components by Apache Beam
"""

import os
from typing import (
    Literal,
    List,
    Dict,
    Any,
    Union,
    Generator,
    Callable,
    get_type_hints,
)
from dataclasses import dataclass, asdict, fields
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
        init_fn: Union[str, Callable[[], Dict]] = None,
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
        if self.init_fn:  # run initi_fn if it is available
            self.config["weak_ref"] = self._shared_handle.acquire(
                _initialize
            )

    def process(
        self, element
    ) -> Generator[
        Union[pd.DataFrame, pd.Series, List[Dict], Dict], None, None
    ]:
        print("Calling processing func")
        # Call the processing function
        outputs = self.processing_fn(element, config=self.config)

        yield outputs


# %%
@dataclass
class BaseData:
    def as_dict(self) -> dict:
        # replace None with string "None"
        out = {
            k: "None" if v is None else v
            for k, v in asdict(self).items()
        }
        out["__class__"] = str(self.__class__)
        return out

    @classmethod
    def from_dict(cls, attrs: dict):
        instance = cls()
        if "__class__" in attrs:
            attrs.pop("__class__")

        type_hints = get_type_hints(cls)

        for field in fields(cls):
            k = field.name
            if k in attrs:
                v = attrs[k]
                if v == "None":
                    setattr(instance, k, None)
                else:
                    field_type = type_hints.get(k, Any)
                    try:
                        # Handle special cases like List, Dict, etc.
                        if hasattr(field_type, "__origin__"):
                            if field_type.__origin__ is list:
                                v = [
                                    field_type.__args__[0](item)
                                    for item in v
                                ]
                            elif field_type.__origin__ is dict:
                                key_type, val_type = field_type.__args__
                                v = {
                                    key_type(key): val_type(value)
                                    for key, value in v.items()
                                }
                        else:
                            v = field_type(v)
                    except ValueError:
                        # If conversion fails, keep the original value
                        pass
                    setattr(instance, k, v)
        return instance

    @staticmethod
    def get_class(class_path: str):
        class_path = class_path.split("'")[1]
        module_path, class_name = class_path.rsplit(".", 1)
        # Import the module
        module = importlib.import_module(module_path)
        # Get the class from the module
        DataClass = getattr(module, class_name)
        return DataClass

    @classmethod
    def has_field(cls, field_name):
        return field_name in cls.__annotations__


@dataclass
class BaseInputData(BaseData):
    batch_size: int = None
    format: Literal["dict", "dataframe"] = "dict"


@dataclass
class BaseOutputData(BaseData):
    pass


# %% CSV Reader and Writers
@dataclass
class CsvInputData(BaseInputData):
    file: str = None
    columns: List[str] = None  # list of columns to read


@dataclass
class CsvOutputData(BaseOutputData):
    file: str = None
    num_shards: int = 1  # csv shards
    headers: List[str] = None  # list of output headers
    compression_type: str = CompressionTypes.UNCOMPRESSED


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
    sql: str = None  # Input sql string
    gcp_project_id: str = (
        None,
    )  # GCP project to run the sql query from
    temp_dataset: str = None  # project_id.temp_dataset


@dataclass(frozen=True, slots=True)
class BigQuerySchemaField:
    name: str
    type: Literal[
        "STRING",
        "TIMESTAMP",
        "INT64",
        "FLOAT64",
        "STRUCT",
        "JSON",
        "BOOL",
        "BYTES",
        "NUMERIC",
        "INTERVAL",
        "DATE",
        "DATETIME",
        "TIME",
        "ARRAY",
        "GEOGRAPHY",
    ]
    mode: Literal["NULLABLE", "REQUIRED", "REPEATED"]
    description: str = ""

    def as_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "type": self.type,
            "mode": self.mode,
            "description": self.description,
        }


@dataclass
class BigQueryOutputData(BaseOutputData):
    output_table: str = None  # project_id.dataset.table
    mode: BigQueryDisposition = (
        BigQueryDisposition.WRITE_APPEND
    )  # BigQuery write mode
    schema: Union[List[BigQuerySchemaField], List[Dict]] = None
    write_method: Literal[
        "FILE_LOADS",
        "STORAGE_WRITE_API",
        "STREAMING_INSERTS",
        "DEFAULT",
    ] = "DEFAULT"

    def __post_init__(self):
        if self.schema and isinstance(
            self.schema[0], BigQuerySchemaField
        ):
            self.schema = {"fields": [s.as_dict() for s in self.schema]}
        elif self.schema and isinstance(self.schema[0], dict):
            # Validating
            assert (
                "name" in self.schema[0]
            ), "'name' must be present in BigQuery Schema"
            assert (
                "type" in self.schema[0]
            ), "'type' must be present in BigQuery Schema"
            assert (
                "mode" in self.schema[0]
            ), "'mode' must be present in BigQuery Schema"
            self.schema = {"fields": self.schema}


class ReadBigQueryData(beam.PTransform):
    def __init__(
        self,
        query: str,
        gcp_project_id: str,
        format: Literal["dataframe", "dict"] = "dict",
        min_batch_size: int = None,
        max_batch_size: int = 1024,
        use_standard_sql: bool = True,
        **kwargs,
    ):
        super().__init__()
        self.query = query
        self.gcp_project_id = gcp_project_id
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
            project=self.gcp_project_id,
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


# %% TFRecords

# %% Parquet



# %% Create the processing function
def _check_gcs_project_id(input_data_gcp_project_id: str, beam_pipeline_args: List[str]):
    if input_data_gcp_project_id:
        return input_data_gcp_project_id

    # Check beam pipeline args
    for arg in beam_pipeline_args:
        if arg.startswith("--projects="):
            return arg.split("=")[1]
    
    # Check environment
    gcp_project_id = os.environ.get("GCP_PROEJCT_ID")
    assert (
        gcp_project_id is not None and gcp_project_id != ""
    ), (
        "GCP Project ID is needed to determine which environment "
        "the SQL query is running in"
    )
    return gcp_project_id


def beam_data_processing_fn(
    input_data: BaseInputData,
    output_data: BaseOutputData,
    processing_fn: str,
    init_fn: str = None,
    beam_pipeline_args: List[str] = ["--runner=DirectRunner"],
) -> None:
    options = PipelineOptions(flags=beam_pipeline_args)

    batch_size = (
        int(input_data.batch_size)
        if input_data.batch_size is not None
        else None
    )
    # Create beam pipeline
    with beam.Pipeline(options=options) as p:
        # inputs
        if isinstance(input_data, CsvInputData):
            pcoll = p | "Read CSV" >> ReadCsvData(
                input_data.file,
                format=input_data.format,
                min_batch_size=batch_size,
            )
        elif isinstance(input_data, BigQueryInputData):
            # Check if --temp_location exists
            assert any(
                [
                    a.startswith("--temp_location=")
                    for a in beam_pipeline_args
                ]
            ), "Need to specify --temp_location argument when reading from BigQuery"
            # Check if gcp_project_id exists
            gcp_project_id = _check_gcs_project_id(input_data.gcp_project_id, beam_pipeline_args)
            pcoll = p | "Read BigQuery" >> ReadBigQueryData(
                query=input_data.sql,
                gcp_project_id=gcp_project_id,
                format=input_data.format,
                min_batch_size=batch_size,
                temp_dataset=input_data.temp_dataset,
            )

        # Run the processing function
        pcoll = pcoll | "Process Data" >> beam.ParDo(
            DataProcessingDoFn(
                processing_fn=processing_fn,
                init_fn=init_fn,
            )
        )

        # Output
        if isinstance(output_data, CsvOutputData):
            num_shards = (
                int(output_data.num_shards)
                if output_data.num_shards is not None
                else None
            )
            pcoll = pcoll | "Write CSV" >> WriteCsvData(
                output_data.file,
                num_shards=num_shards,
                headers=output_data.headers,
            )
        elif isinstance(output_data, BigQueryOutputData):
            assert any(
                [
                    a.startswith("--temp_location=")
                    for a in beam_pipeline_args
                ]
            ), "Need to specify --temp_location argument when writing to BigQuery"
            pcoll = pcoll | "Write BigQuery" >> WriteBigQueryData(
                output_table=output_data.output_table,
                schema=output_data.schema,
                write_disposition=output_data.mode,
                method=output_data.write_method,
            )
