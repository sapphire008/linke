"""Living as an independent file so that it can import the helper functions from another file"""

from kfp import dsl
from typing import List, Dict

from kubeflow_components.dataset.beam_data_processor import BaseInputData, BaseOutputData


@dsl.component
def beam_data_processing_component(
    processing_fn: str,
    init_fn: str,
    input_data: Dict,
    output_data: Dict,
    # input_artifact: Optional[Input[Artifact]] = None,
    # output_artifact: Optional[Output[Artifact]] = None,
    beam_pipeline_args: List[str] = ["--runner=DirectRunner"],
):
    """Beam data processing Kubeflow component."""
    # Importing all the helper functions
    from kubeflow_components.dataset.beam_data_processor import (
        beam_data_processing_fn,
        BaseData,
    )
    
    # Getting the data class
    input_dataclass = BaseData.get_class(input_data["__class__"])
    output_dataclass = BaseData.get_class(output_data["__class__"])

    # If artifacts are specified
    # if hasattr(input_data, "file"):
    #     if input_artifact is not None:
    #         input_data.file = input_artifact.path
    #     else:
    #         assert input_data.file is not None, (
    #             "Either specify an input_artifact, or specify "
    #             "the `file` attribute in input_data"
    #         )
    # if hasattr(output_data, "file"):
    #     if output_artifact is not None:
    #         output_data.file = output_artifact.path
    #     else:
    #         assert output_data.file is not None, (
    #             "Either specify an output_artifact, or specify "
    #             "the `file` attribute in output_data"
    #         )

    # Call the data processor
    beam_data_processing_fn(
        input_data = input_dataclass.from_dict(input_data),
        output_data=output_dataclass.from_dict(output_data),
        processing_fn=processing_fn,
        init_fn=init_fn,
        beam_pipeline_args=beam_pipeline_args,
    )
