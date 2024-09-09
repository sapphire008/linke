"""Archive file data sources"""
from typing import Optional
import apache_beam as beam
from apache_beam.io import iobase, filebasedsource, filebasedsink
from apache_beam.io.filesystem import CompressionTypes
from apache_beam.coders import coders



class ArchiveType:
    """Enum class for a list of archive type"""
    TAR = "tar"
    ZIP = "zip"


# %% Read
class _ArchiveFileSource(filebasedsource.FileBasedSource):
    def __init__(
        self,
        archive_type: str = ArchiveType.TAR,
        compression_type: str = CompressionTypes.AUTO,
    ):
        pass


# %% Write
class _ArchiveFileSink(filebasedsink.FileBasedSink):
    """A sink for archive files"""

    def __init__(
        self,
        file_path_prefix: str,
        file_name_suffix: str='',
        archive_type: str = ArchiveType.TAR,
        compression_type: str = CompressionTypes.AUTO,
        coder: coders.Coder = coders.ToBytesCoder(),
        num_shards: int = 0,
        shard_name_template: str = None,
        max_records_per_shard: Optional[int] =None,
        max_bytes_per_shard: Optional[int]=None,
        skip_if_empty: bool = True,
    ):
        super().__init__(
            file_path_prefix,
            file_name_suffix=file_name_suffix,
            num_shards=num_shards,
            shard_name_template=shard_name_template,
            coder=coder,
            mime_type='application/json',
            compression_type=compression_type,
            max_records_per_shard=max_records_per_shard,
            max_bytes_per_shard=max_bytes_per_shard,
            skip_if_empty=skip_if_empty,
        )
        self._archive_type = archive_type

    def open(self, temp_path):
        file_handle = super().open(temp_path)
        return file_handle

    def close(self, file_handle):
        super().close(file_handle)

    def display_data(self):
        dd_parent = super().display_data()
        return dd_parent

    def write_encoded_record(self, file_handle, encoded_value):
        """Writes a single encoded record."""
        file_handle.write(encoded_value)


if __name__ == '__main__':
    data = {
        "row": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
        "A": list("acbdgsftvagadmzk"),
        "B": [0.6, 0.2, 0.3, 0.7, 0.1, 0.3, 0.5, 0.2, 0.6, 0.3, 0.5, 0.9, 0.1, 0.4, 0.5, 0.6],
        "C": [True, False, False, True, True, True, False, False, True, False, True, True, True, False, True, False],
        "D": [[1, 3], [2], [6, 2, 3], [], [], [], [2, 5], [1], [], [3, 5], [3, 4], [4, 5], [4], [], [1, 9], [8]],   
    }
    data = [dict(zip(data,t)) for t in zip(*data.values())]
    