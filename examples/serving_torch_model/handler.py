"""
ModelHandler defines a custom model handler.
Initialization and serving logics are implemented here.
"""

import os
import logging
from typing import Dict, Any, List
import torch
from ts.torch_handler.base_handler import BaseHandler
from ts.context import Context as TorchServeContext

# This import is okay because the hanlder.py file will be
# in ths same directory as the model.py file after packaging.
from model import MaskNet

logger = logging.getLogger(__name__)

class ModelHandler(BaseHandler):
    """
    A custom PyTorch model handler implementation.
    """

    def __init__(self, *args, **kwargs):
        super(ModelHandler, self).__init__()
        self._context = None
        self.initialized = False

    def initialize(self, context: TorchServeContext):
        """
        Initialize model. This will be called during model loading time
        :param context: Initial context contains model server system properties.
        :return:
        """
        self._context = context
        self.manifest = context.manifest
        properties = context.system_properties

        # Set inference devices
        self.device = torch.device(
            "cuda:" + str(properties.get("gpu_id"))
            if torch.cuda.is_available()
            and properties.get("gpu_id") is not None
            else "cpu"
        )

        model_dir = properties.get("model_dir")

        # Read model config file
        model_config_path = os.path.join(model_dir, "model_config.yaml")

        # Read model serialize/pt file
        serialized_file = self.manifest["model"]["serializedFile"]
        model_weight_path = os.path.join(model_dir, serialized_file)

        # defining and loading the custom model
        self.model = MaskNet.from_config(model_config_path)
        self.model.to(self.device)

        # Load the weights
        model_weights = torch.load(model_weight_path)
        self.model.load_state_dict(model_weights)
        self.model.eval()  # set to eval model
        
        projection_weight = model_weights["output_projection.parallel_output_projection_0.weight"]
        logger.info(f"Loaded model weights {projection_weight}")
        acquired_projection_weight = self.model.output_projection.parallel_output_projection_0.weight
        logger.info(f"Current model weights {acquired_projection_weight}")
        assert (projection_weight == acquired_projection_weight).all().tolist(), "Not all weights are equal after loading"

        self.initialized = True

    def preprocess(self, data: List):
        """
        Transform raw input into model input data.
        :param batch: list of raw requests, should match batch size
        :return: list of preprocessed model input data
        """
        # Take the input data and make it inference ready
        preprocessed_data = data[0].get("data")
        if preprocessed_data is None:
            preprocessed_data = data[0].get("body")
        # Convert to tensor and ship to device
        for k, v in preprocessed_data.items():
            preprocessed_data[k] = torch.tensor(v).to(self.device)

        return preprocessed_data

    def inference(self, inputs: Dict):
        """
        Internal inference methods
        :param model_input: transformed model input data
        :return: list of inference output in NDArray
        """
        # Do some inference call to engine here and return output
        with torch.no_grad():
            model_output = self.model(inputs)
        return model_output

    def postprocess(self, inference_output):
        """
        Return inference result.
        :param inference_output: list of inference output
        :return: list of predict results
        """
        # Take output from network and post-process to desired format
        postprocess_output = inference_output.tolist() # tensor -> list
        return postprocess_output

    def handle(
        self,
        data: List[Dict[str, bytearray]],
        context: TorchServeContext,
    ):
        """
        Invoked by TorchServe for prediction request.
        Do pre-processing of data, prediction using model and postprocessing of prediciton output
        :param data: Input data for prediction
        :param context: Initial context contains model server system properties.
        :return: prediction output
        """
        if not self.initialized:
            self.initialized(context)
        model_input = self.preprocess(data)
        model_output = self.inference(model_input)        
        model_output = self.postprocess(model_output)
        logger.info("model output: " + str(model_output))
        # Batch size needs to match the expected batch size of the torchserve_config.yaml
        # Python objects only. Need to cast tensors to lists
        return model_output
