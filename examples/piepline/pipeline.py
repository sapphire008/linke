
from dynaconf import Dynaconf
from kfp import dsl


@dsl.pipeline
def create_pipeline_components(settings: Dynaconf):
    pass


