import os
import re
import tempfile
import shutil
import importlib
from typing import Union, Dict, Any, Optional
from kfp.dsl.base_component import BaseComponent
from kfp import compiler
from google.cloud import aiplatform


class VertexPipelineRunner:
    def __init__(self, gcp_project_id: str, run_region: str):
        aiplatform.init(project=gcp_project_id, location=run_region)

    def sanitize_pipeline_name(self, pipeline_name):
        # Convert to lowercase
        sanitized = pipeline_name.lower()

        # Replace any character that's not a-z, 0-9, or - with a hyphen
        sanitized = re.sub(r"[^a-z0-9-]", "-", sanitized)

        # Remove any leading hyphens
        sanitized = sanitized.lstrip("-")

        # Ensure it starts with a letter or number if it's empty after previous operations
        if not sanitized:
            sanitized = "pipeline"
        elif not sanitized[0].isalnum():
            sanitized = "p" + sanitized

        # Truncate to 128 characters
        sanitized = sanitized[:128]

        return sanitized

    def compile_pipeline_package(
        self,
        pipeline: Union[str, BaseComponent],
        pipeline_name: str,
        pipeline_path: Optional[str] = None,
    ):
        # Compile the pipeline_func
        if isinstance(pipeline, str):  # import the pipeline from module
            if (
                pipeline.endswith(".yaml") or pipeline.endswith(".yml")
            ) and os.path.isfile(pipeline):
                self.pipeline = pipeline  # existing file
                return
            else:  # assuming path to the pipeline function
                module_path, function_name = pipeline.rsplit(".", 1)
                # Import the module
                module = importlib.import_module(module_path)
                # Get the function from the module
                pipeline = getattr(module, function_name)

        # Compile BaseComponent
        temp_dir = None
        if pipeline_path is None:
            temp_dir = tempfile.mkdtemp()
            pipeline_path = os.path.join(
                temp_dir, f"{pipeline_name}.yaml"
            )
        else:
            pipeline_path = os.path.join(
                pipeline_path, f"{pipeline_name}.yaml"
            )
        self.pipeline = pipeline_path
        compiler.Compiler().compile(
            pipeline_func=pipeline,
            package_path=pipeline_path,
            pipeline_name=pipeline_name,
        )
        return temp_dir

    def create_run(
        self,
        pipeline: Union[str, BaseComponent],
        pipeline_name: str,
        pipeline_root: str,
        pipeline_parameters: Dict[str, Any] = {},
        pipeline_path: Optional[str] = None,
        enable_caching: bool = False,
        image_uri: str = None,
    ):
        """
        Create a run and submit to GCP Vertex AI

        Parameters
        ----------
        pipeline: one of the following 3:
            - @dsl.pipeline decorated pipeline function
            - Import path to the @dsl.pipeline decoration pipeline function, separated by "."
            - .yaml file compiled from the pipeline function
        pipeline_name : str
            Displayed name of the pipeline.
        pipeline_root : str
            Remote storage path to store pipeline output artifacts
        pipeline_parameters : Dict[str, Any], optional
            Input parameters of the pipeline, by default {}
        pipeline_path : Optional[str], optional
            A directory to store the compiled pipeline.yaml file.
            Only used when `pipeline` argument is a BaseComponent
            @dsl.pipeline decorated function, or the import path of
            the fnuction. By default, None, which will save the
            compiled .yaml file to a temporary directory.
        enable_caching : bool, optional
            Whether or not to enable run caching, by default False
        image_uri : str, optional
            Docker image path to use to run the pipeline, by default None
        """
        # Clean the display name
        pipeline_name = self.sanitize_pipeline_name(pipeline_name)
        temp_dir = self.compile_pipeline_package(
            pipeline, pipeline_name, pipeline_path
        )

        # Create the job
        job = aiplatform.PipelineJob(
            display_name=pipeline_name,
            template_path=self.pipeline,
            # job_id="test-kfp2-vertex-run",
            pipeline_root=pipeline_root,
            parameter_values=pipeline_parameters,
            enable_caching=enable_caching,
            # encryption_spec_key_name = CMEK,
            # labels = LABELS,
            # credentials = CREDENTIALS,
            # failure_policy = FAILURE_POLICY
        )

        # Submit the job
        job.submit()
        
        # Clean up
        if temp_dir:
            shutil.rmtree(temp_dir)
