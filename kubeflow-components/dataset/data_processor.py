"""
Apache Beam based data procesor. The overall architecture is

Reader -> Custom processor class -> Writer

Reader and Writer are a set of supported components by Apache Beam
"""

import apache_beam as beam


