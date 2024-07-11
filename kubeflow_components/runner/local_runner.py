import tempfile
from kfp import dsl, local


class LocalDagRunner:
    def __init__(self, pipeline_root: str = None, runner=None):
        # Initialize the dag runner
        self.pipeline_root = pipeline_root or tempfile.mkdtemp()
        local.init(
            runner=local.SubprocessRunner(use_venv=False),
            pipeline_root=self.pipeline_root,
            raise_on_error=True,
        )
        
    def run(self, pipeline: dsl.graph_component.GraphComponent, payload: dict =None):
        payload = payload or {}
        # Run by directlycalling the pipeline function
        result= pipeline(**payload)
        # Return the result
        return result

