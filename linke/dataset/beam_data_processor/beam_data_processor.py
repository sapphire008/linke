"""
Apache Beam based data procesor. The overall architecture is:

Reader -> Custom processor class -> Writer

Reader and Writer are a set of supported components by Apache Beam
"""

import os
# fmt: off
from typing import (
    List, Dict, Any, Type, Tuple, Literal, Optional, Union,
    Generator, Callable, get_type_hints,
)
# fmt: on
import json
from dataclasses import dataclass, asdict, fields
import keyword
import importlib
import numpy as np
import pandas as pd
import pyarrow as pa
import apache_beam as beam
from apache_beam.options.pipeline_options import PipelineOptions
from apache_beam.io.filesystem import CompressionTypes
from apache_beam.io.textio import ReadFromCsv, ReadFromText, WriteToText
from apache_beam.io.gcp.bigquery import (
    ReadFromBigQuery,
    WriteToBigQuery,
    BigQueryDisposition,
)
from apache_beam.io.tfrecordio import ReadFromTFRecord, WriteToTFRecord
from apache_beam.io.parquetio import ReadFromParquet, WriteToParquet

from kubeflow_components.dataset.beam_data_processor.utils import (
    TFRecordIOUtils,
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
    def import_function(path: str) -> Callable:
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

    @staticmethod
    def dict2list(inputs: Union[Dict, pd.DataFrame]) -> List[Dict]:
        """Assuming values are arrays of the same length."""
        import pandas as pd

        try:
            df_inputs = pd.DataFrame(inputs)
        except:
            raise (
                ValueError(
                    "Failed the attempt to convert from dict of features  "
                    "to records. Features may not be the same length. "
                    "Check processing_fn output implementation"
                )
            )
        return df_inputs.to_dict("records")

    def process(
        self, element
    ) -> Generator[
        Union[pd.DataFrame, pd.Series, List[Dict], Dict], None, None
    ]:
        # Call the processing function
        outputs = self.processing_fn(element, config=self.config)

        # Convert to list of dict iff returning dict
        if not isinstance(outputs, (list, tuple)):
            outputs = self.dict2list(outputs)

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
    # fmt: off
    type: Literal[
        "STRING", "TIMESTAMP", "INT64", "FLOAT64", "STRUCT",
        "JSON", "BOOL", "BYTES", "NUMERIC", "INTERVAL", "DATE",
        "DATETIME", "TIME", "ARRAY", "GEOGRAPHY",
    ]
    # fmt: on
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
            pcoll = pcoll | "Convert Format" >> beam.Map(
                self._convert_to_df
            )

        return pcoll


class WriteBigQueryData(beam.PTransform):
    def __init__(
        self,
        output_table: str,
        schema: Dict,
        write_disposition: BigQueryDisposition = BigQueryDisposition.WRITE_APPEND,
        is_batched: bool = False,
        **kwargs,
    ):
        self.table = self.normalize_table_reference(output_table)
        self.schema = schema  # bigquery specific schema
        self.write_disposition = write_disposition
        self.is_batched = is_batched
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
        if self.is_batched:  # unbatching
            pcoll = pcoll | "Unbatching" >> beam.FlatMap(lambda x: x)

        return pcoll | "Write to BigQuery" >> WriteToBigQuery(
            table=self.table,
            schema=self.schema,
            write_disposition=self.write_disposition,
            **self.kwargs,
        )


# %% TFRecords
@dataclass(frozen=True, slots=True)
class TFRecordFeatureSchema:
    name: str
    type: Literal["int", "float", "byte"]
    fixed_length: bool = True


@dataclass
class TFRecordInputData(BaseInputData):
    file: str = None
    format: Literal["dict", "dataframe", "feature"] = "feature"
    schema: List[TFRecordFeatureSchema] = None
    compression_type: str = CompressionTypes.GZIP
    deserialize_data: bool = True

    def __post_init__(self):
        # Converting schema to dict
        schema = {}
        feature_type = {}
        for field in self.schema:
            schema[field.name] = field.type
            feature_type[field.name] = (
                "fixed" if field.fixed_length else "variable"
            )
        self.schema = schema
        self.feature_type = feature_type


@dataclass
class TFRecordOutputData(BaseOutputData):
    file: str = None
    schema: List[TFRecordFeatureSchema] = None
    compression_type: str = CompressionTypes.GZIP
    serialize_data: bool = True
    num_shards: int = 0
    shard_name_template: str = ""

    def __post_init__(self):
        # Converting schema to dict
        schema = {}
        feature_type = {}
        for field in self.schema:
            schema[field.name] = field.type
            feature_type[field.name] = (
                "fixed" if field.fixed_length else "variable"
            )
        self.schema = schema
        self.feature_type = feature_type


class ReadTFRecordData(beam.PTransform):
    def __init__(
        self,
        file_pattern,
        schema: Dict[str, Literal["byte", "int", "float"]],
        feature_type: Dict[str, Literal["fixed", "variable"]],
        compression_type="GZIP",
        deserialize: bool = True,
        format: Literal["dataframe", "dict", "feature"] = "feature",
        min_batch_size: int = None,
        max_batch_size: int = 1024,
    ):
        self.file_pattern = file_pattern
        self.schema = schema
        self.feature_type = feature_type
        self.compression_type = compression_type
        self.deserialize = deserialize
        self.format = format
        self.min_batch_size = min_batch_size
        self.max_batch_size = max_batch_size

    def _convert_result(self, x: Union[Dict, List[Dict]]):
        import pandas as pd

        if self.min_batch_size is None:
            return pd.Series(x)  # pd.Series
        else:
            df = pd.DataFrame(x)
            if self.format == "feature":
                # Check if all arrays have the same length
                # Return a dict of batched features
                df = df.to_dict("list")
                # Stacking those fixed length list features
                df = {
                    k: (
                        np.stack(v)
                        if self.feature_type[k] == "fixed"
                        else v
                    )
                    for k, v in df.items()
                }
            return df

    def expand(self, pcoll):
        pcoll = pcoll | "Read TFRecord Data" >> ReadFromTFRecord(
            file_pattern=self.file_pattern,
            compression_type=self.compression_type,
        )

        # Deserialization
        if self.deserialize:
            pcoll = pcoll | "Deserialize" >> beam.Map(
                lambda x: TFRecordIOUtils.deserialize_tf_example(
                    x, self.schema
                )
            )

        # Batching:
        if self.min_batch_size is not None:
            pcoll = pcoll | "Batching" >> beam.BatchElements(
                min_batch_size=self.min_batch_size,
                max_batch_size=self.max_batch_size,
            )

        # Optionally convert to dataframe if deserialized
        if self.deserialize and (
            self.format == "dataframe"
            or (
                self.format == "feature"
                and self.min_batch_size is not None
            )
        ):
            pcoll = pcoll | "Convert Format" >> beam.Map(
                self._convert_result
            )
        return pcoll


class WriteTFRecordsData(beam.PTransform):
    def __init__(
        self,
        file_path: str,
        schema: Dict[str, Literal["byte", "int", "float"]],
        is_batched: bool = True,
        serialize_data: bool = True,
        num_shards: int = 0,
        shard_name_template: str = "",
        compression_type: str = CompressionTypes.GZIP,
    ):
        filepath, filename = os.path.dirname(
            file_path
        ), os.path.basename(file_path)
        filename, fileext = os.path.splitext(filename)
        self.filepath_prefix = os.path.join(filepath, filename)
        self.filepath_suffix = fileext
        self.schema = schema
        self.num_shards = num_shards
        self.is_batched = is_batched
        self.serialize_data = serialize_data
        self.shard_name_template = shard_name_template
        self.compression_type = compression_type

    def expand(self, pcoll):
        if self.is_batched:
            pcoll = pcoll | "Unbatch" >> beam.FlatMap(lambda x: x)
        if self.serialize_data:
            pcoll = pcoll | "Serialize Data" >> beam.Map(
                lambda x: TFRecordIOUtils.serialize_tf_example(
                    x, self.schema
                )
            )
        return pcoll | "Write to TFRecord" >> WriteToTFRecord(
            file_path_prefix=self.filepath_prefix,
            file_name_suffix=self.filepath_suffix,
            num_shards=self.num_shards,
            shard_name_template=self.shard_name_template,
            compression_type=self.compression_type,
        )


# %% Parquet
@dataclass(slots=True)
class ParquetSchemaField:
    """Provides better dtype annotation and than pa.schema"""

    name: str
    # fmt: off
    type: Union[Union[pa.DataType,
        Literal[
            "string", "int", "float", "bool", 
            "timestamp", # UTC timestamp in seconds
            "array(string)", "array(int)", "array(float)", 
            "array(bool)", "array(timestamp)",
    ],],]
    # fmt: on
    nullable: bool = True
    metadata: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        if not isinstance(self.type, str):
            return
        str2dtype = {
            "string": pa.string(),
            "int": pa.int64(),
            "float": pa.float32(),
            "bool": pa.bool_(),
            "timestamp": pa.timestamp("s", tz="UTC"),
        }
        # map from string to pa.DataType
        dtype = self.type.replace(" ", "")  # replace any space
        if dtype.startswith("array"):
            dtype = self.type.replace("array(", "").replace(")", "")
            assert dtype in str2dtype, f"Unrecognized dtype {self.type}"
            self.type = pa.list_(str2dtype[dtype])
        else:
            assert dtype in str2dtype, f"Unrecognized dtype {self.type}"
            self.type = str2dtype[dtype]

    def as_field(self):
        return pa.field(
            self.name,
            self.type,
            nullable=self.nullable,
            metadata=self.metadata,
        )


@dataclass
class ParquetInputData(BaseInputData):
    file: str = None
    columns: List[str] = None  # subset of columns


@dataclass
class ParquetOutputData(BaseOutputData):
    file: str = None
    schema: Union[List[ParquetSchemaField], pa.Schema] = None
    # only used when schema is a List[ParquetSchemaField]
    schema_metadata: Optional[Dict[str, str]] = None
    num_shards: int = 0
    shard_name_template: str = ""

    def __post_init__(self):
        # Converting parqeut schema to pyarrow.Schema
        if isinstance(self.schema, list) and isinstance(
            self.schema[0], ParquetSchemaField
        ):
            self.schema = pa.schema([f.as_field() for f in self.schema])
            if self.schema_metadata:
                self.schema = self.schema.with_metadata(
                    self.schema_metadata
                )


class ReadParquetData(beam.PTransform):
    def __init__(
        self,
        file_pattern,
        columns: List[str] = None,
        format: Literal["dataframe", "dict"] = "dict",
        min_batch_size: int = None,
        max_batch_size: int = 1024,
        **kwargs,
    ):
        self.file_pattern = file_pattern
        self.columns = columns
        self.format = format
        self.min_batch_size = min_batch_size
        self.max_batch_size = max_batch_size
        self.kwargs = kwargs

    def _convert_to_df(self, x: Union[Dict, List[Dict]]):
        import pandas as pd

        if self.min_batch_size is None:
            return pd.Series(x)  # pd.Series
        else:
            return pd.DataFrame(x)

    def expand(self, pcoll):
        pcoll = pcoll | "Read Parquet Data" >> ReadFromParquet(
            file_pattern=self.file_pattern,
            columns=self.columns,
            **self.kwargs,
        )

        # Batching:
        if self.min_batch_size is not None:
            pcoll = pcoll | "Batching" >> beam.BatchElements(
                min_batch_size=self.min_batch_size,
                max_batch_size=self.max_batch_size,
            )

        # Default outputs dictionary
        if self.format == "dataframe":
            pcoll = pcoll | "Convert Format" >> beam.Map(
                self._convert_to_df
            )
        return pcoll


class WriteParquetData(beam.PTransform):
    def __init__(
        self,
        file_path: str,
        schema: Optional[pa.Schema] = None,
        is_batched: bool = True,
        num_shards: int = 0,
        shard_name_template: str = "",
        **kwargs,
    ):
        filepath, filename = os.path.dirname(
            file_path
        ), os.path.basename(file_path)
        filename, fileext = os.path.splitext(filename)
        self.filepath_prefix = os.path.join(filepath, filename)
        self.filepath_suffix = fileext
        self.schema = schema
        self.num_shards = num_shards
        self.is_batched = is_batched
        self.shard_name_template = shard_name_template
        self.kwargs = kwargs

    def expand(self, pcoll):
        if self.is_batched:
            pcoll = pcoll | "Unbatch" >> beam.FlatMap(lambda x: x)

        # return pcoll | beam.Map(print)

        return pcoll | "Write to Parquet" >> WriteToParquet(
            file_path_prefix=self.filepath_prefix,
            file_name_suffix=self.filepath_suffix,
            schema=self.schema,
            num_shards=self.num_shards,
            shard_name_template=self.shard_name_template,
            **self.kwargs,
        )


# %% WebDataset (Tar files)
class ReadWebDatasetData(beam.PTransform):
    def __init__(
        self,
        file_pattern,
        columns: List[str] = None,
        format: Literal["dataframe", "dict"] = "dict",
        min_batch_size: int = None,
        max_batch_size: int = 1024,
        **kwargs,
    ):
        self.file_pattern = file_pattern
        self.columns = columns
        self.format = format
        self.min_batch_size = min_batch_size
        self.max_batch_size = max_batch_size
        self.kwargs = kwargs

    def _convert_to_df(self, x: Union[Dict, List[Dict]]):
        import pandas as pd

        if self.min_batch_size is None:
            return pd.Series(x)  # pd.Series
        else:
            return pd.DataFrame(x)

    def expand(self, pcoll):
        pcoll = pcoll | "Read Parquet Data" >> ReadFromParquet(
            file_pattern=self.file_pattern,
            columns=self.columns,
            **self.kwargs,
        )

        # Batching:
        if self.min_batch_size is not None:
            pcoll = pcoll | "Batching" >> beam.BatchElements(
                min_batch_size=self.min_batch_size,
                max_batch_size=self.max_batch_size,
            )

        # Default outputs dictionary
        if self.format == "dataframe":
            pcoll = pcoll | "Convert Format" >> beam.Map(
                self._convert_to_df
            )
        return pcoll


class WriteWebDatasetData(beam.PTransform):
    def __init__(
        self,
        file_path: str,
        schema: Optional[pa.Schema] = None,
        is_batched: bool = True,
        num_shards: int = 0,
        shard_name_template: str = "",
        **kwargs,
    ):
        filepath, filename = os.path.dirname(
            file_path
        ), os.path.basename(file_path)
        filename, fileext = os.path.splitext(filename)
        self.filepath_prefix = os.path.join(filepath, filename)
        self.filepath_suffix = fileext
        self.schema = schema
        self.num_shards = num_shards
        self.is_batched = is_batched
        self.shard_name_template = shard_name_template
        self.kwargs = kwargs

    def expand(self, pcoll):
        if self.is_batched:
            pcoll = pcoll | "Unbatch" >> beam.FlatMap(lambda x: x)

        # return pcoll | beam.Map(print)

        return pcoll | "Write to Parquet" >> WriteToParquet(
            file_path_prefix=self.filepath_prefix,
            file_name_suffix=self.filepath_suffix,
            schema=self.schema,
            num_shards=self.num_shards,
            shard_name_template=self.shard_name_template,
            **self.kwargs,
        )


# %% Create the processing function
def _check_gcs_project_id(
    input_data_gcp_project_id: str, beam_pipeline_args: List[str]
):
    if input_data_gcp_project_id:
        return input_data_gcp_project_id

    # Check beam pipeline args
    for arg in beam_pipeline_args:
        if arg.startswith("--projects="):
            return arg.split("=")[1]

    # Check environment
    gcp_project_id = os.environ.get("GCP_PROEJCT_ID")
    assert gcp_project_id is not None and gcp_project_id != "", (
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
            gcp_project_id = _check_gcs_project_id(
                input_data.gcp_project_id, beam_pipeline_args
            )
            pcoll = p | "Read BigQuery" >> ReadBigQueryData(
                query=input_data.sql,
                gcp_project_id=gcp_project_id,
                format=input_data.format,
                min_batch_size=batch_size,
                temp_dataset=input_data.temp_dataset,
            )
        elif isinstance(input_data, TFRecordInputData):
            pcoll = p | "Read TFRecord" >> ReadTFRecordData(
                file_pattern=input_data.file,
                schema=input_data.schema,
                feature_type=input_data.feature_type,
                compression_type=input_data.compression_type,
                format=input_data.format,
                min_batch_size=input_data.batch_size,
            )
        elif isinstance(input_data, ParquetInputData):
            pcoll = p | "Read Parquet" >> ReadParquetData(
                file_pattern=input_data.file,
                columns=input_data.columns,
                format=input_data.format,
                min_batch_size=input_data.batch_size,
            )

        # # Run the processing function
        pcoll = pcoll | "Process Data" >> beam.ParDo(
            DataProcessingDoFn(
                processing_fn=processing_fn,
                init_fn=init_fn,
            )
        )

        # return pcoll | beam.Map(print)

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
                is_batched=batch_size is not None,
            )
        elif isinstance(output_data, TFRecordOutputData):
            pcoll = pcoll | "Write TFRecords" >> WriteTFRecordsData(
                file_path=output_data.file,
                schema=output_data.schema,
                is_batched=batch_size is not None,
                serialize_data=output_data.serialize_data,
                num_shards=output_data.num_shards,
                shard_name_template=output_data.shard_name_template,
            )
        elif isinstance(output_data, ParquetOutputData):
            pcoll = pcoll | "Write Parquet" >> WriteParquetData(
                file_path=output_data.file,
                schema=output_data.schema,
                is_batched=batch_size is not None,
                num_shards=output_data.num_shards,
                shard_name_template=output_data.shard_name_template,
            )
