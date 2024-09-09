"""Archive file data sources"""

import io
import json
from typing import Optional
import tarfile
import uuid
import apache_beam as beam
from apache_beam.io import iobase, filebasedsource, filebasedsink
from apache_beam.io.filesystem import CompressionTypes, FileSystem
from apache_beam.io.filesystems import FileSystems
from apache_beam.coders import coders
from apache_beam.options.value_provider import check_accessible


class ArchiveType:
    """Enum class for a list of archive type"""

    TAR = "tar"
    ZIP = "zip"


class JsonCoder(coders.Coder):
    """A JSON coder interpreting each line as a JSON string."""

    def encode(self, x: dict) -> str:
        return json.dumps(x).encode("utf-8")

    def decode(self, x: str) -> dict:
        return json.loads(x)


# %% Read
class _ArchiveFileSource(filebasedsource.FileBasedSource):
    def __init__(
        self,
        archive_type: str = ArchiveType.TAR,
        compression_type: str = CompressionTypes.AUTO,
        coder: coders.Coder = JsonCoder(),
    ):
        pass


# %% Write
class _ArchiveFileSink(filebasedsink.FileBasedSink):
    """A sink for archive files. Output must be byte serializable."""

    def __init__(
        self,
        file_path_prefix: str,
        file_name_suffix: str = "",
        archive_type: str = ArchiveType.TAR,
        compression_type: str = CompressionTypes.AUTO,
        coder: coders.Coder = JsonCoder(),
        num_shards: int = 0,
        shard_name_template: str = None,
        max_records_per_shard: Optional[int] = None,
        max_bytes_per_shard: Optional[int] = None,
        skip_if_empty: bool = True,
    ):
        super().__init__(
            file_path_prefix,
            file_name_suffix=file_name_suffix,
            num_shards=num_shards,
            shard_name_template=shard_name_template,
            coder=coder,
            mime_type="application/octet-stream",
            compression_type=compression_type,
            max_records_per_shard=max_records_per_shard,
            max_bytes_per_shard=max_bytes_per_shard,
            skip_if_empty=skip_if_empty,
        )
        self._archive_type = archive_type

    @check_accessible(["file_path_prefix"])
    def open(self, temp_path):
        """Opens ``temp_path``, returning an opaque file handle object.

        The returned file handle is passed to ``write_[encoded_]record`` and
        ``close``.
        """
        if self._archive_type == ArchiveType.TAR:
            if self.compression_type in (
                CompressionTypes.AUTO,
                CompressionTypes.UNCOMPRESSED,
            ):  # no compression by default
                mode = "w"
            elif self.compression_type == CompressionTypes.GZIP:
                mode = "w:gz"
            else:
                raise ValueError(
                    "Only uncompressed or gzip compression "
                    "supported for tar archive"
                )
            # This makes sure that we can write to s3 or gcs paths as well
            fobj = FileSystems.create(
                temp_path,
                mime_type="application/octet-stream",
                compression_type=CompressionTypes.UNCOMPRESSED,
            )  # use beam to open write channel
            writer = tarfile.open(mode=mode, fileobj=fobj)
        else:
            raise NotImplementedError(
                f"Archive type {self._archive_type} not implemented."
            )

        if self.max_bytes_per_shard:
            self.byte_counter = filebasedsink._ByteCountingWriter(
                writer
            )
            return self.byte_counter
        else:
            return writer

    def close(self, file_handle):
        super().close(file_handle)

    def write_encoded_record(self, file_handle, encoded_value):
        """Writes a single encoded record."""
        file_like_object = io.BytesIO(encoded_value)

        # Making sure a unique file name per value
        idx = str(uuid.uuid4())
        # Write to file
        if self._archive_type == ArchiveType.TAR:
            tarinfo = tarfile.TarInfo(name=f"json-{idx}.json")
            tarinfo.size = len(encoded_value)
            # Add the file to the tar archive
            file_handle.addfile(tarinfo, fileobj=file_like_object)
        elif self._archive_type == ArchiveType.ZIP:
            pass


class WriteToWebDataSet(beam.PTransform):
    """
    Write to webdataset.
    Data needs to be serializable to bytes.
    Currently only support tabular data from ptransform.
    """
    def __init__(
        self,
        file_path_prefix: str,
        file_name_suffix: str = "",
        compression_type: str = CompressionTypes.AUTO,
        coder: coders.Coder = JsonCoder(),
        num_shards: int = 0,
        shard_name_template: str = None,
        max_records_per_shard: Optional[int] = None,
        max_bytes_per_shard: Optional[int] = None,
    ):
        super().__init__()
        self._sink = _ArchiveFileSink(
            file_path_prefix,
            file_name_suffix=file_name_suffix,
            archive_type=ArchiveType.TAR,
            compression_type=compression_type,
            coder=coder,
            num_shards=num_shards,
            shard_name_template=shard_name_template,
            max_records_per_shard=max_records_per_shard,
            max_bytes_per_shard=max_bytes_per_shard,
            skip_if_empty=True,
        )

    def expand(self, pcoll):
        return pcoll | iobase.Write(self._sink)


if __name__ == "__main__":
    # fmt: off
    df = {
        "row": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
        "A": list("acbdgsftvagadmzk"),
        "B": [0.6, 0.2, 0.3, 0.7, 0.1, 0.3, 0.5, 0.2, 0.6, 0.3, 0.5, 0.9, 0.1, 0.4, 0.5, 0.6],
        "C": [True, False, False, True, True, True, False, False, True, False, True, True, True, False, True, False],
        "D": [[1, 3], [2], [6, 2, 3], [], [], [], [2, 5], [1], [], [3, 5], [3, 4], [4, 5], [4], [], [1, 9], [8]],   
    }
    df = [dict(zip(df,t)) for t in zip(*df.values())]
    # fmt: on
    from apache_beam.testing.test_pipeline import TestPipeline

    with TestPipeline() as p:
        input = (
            p
            | beam.Create(df)
            | WriteToWebDataSet(
                file_path_prefix="./sample_data",
                file_name_suffix=".tgz",
                compression_type=CompressionTypes.GZIP,
                max_records_per_shard=4,
            )
        )
