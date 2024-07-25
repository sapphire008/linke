import sys
import os
import torch
import subprocess
from dataclasses import asdict

base_dir = os.path.abspath(os.path.realpath(
    os.path.join(os.path.dirname(__file__), "../..")
))
if base_dir not in sys.path:
    sys.path.insert(0, base_dir)

sub_dir = "examples/masknet_recommender"

from kubeflow_components.serving.torchserve.save_model import export_to_model_archive
from examples.masknet_recommender.model import MaskNet, FeatureSpec, FeatureEmbedding


# %% Create the untrained model and save weights
feature_specs = {
    "day_of_week": FeatureSpec(
        type="categorical",
        embed_size=10,
        vocab_size=7,
        padding_idx=0,
    ),
    "hour_of_day": FeatureSpec(
        type="categorical", embed_size=10, vocab_size=24
    ),
    "account_tenure": FeatureSpec(
        type="categorical",
        embed_size=10,
        vocab_size=5,
        padding_idx=0,
    ),
    "payment_tier": FeatureSpec(
        type="categorical",
        embed_size=10,
        vocab_size=3,
        padding_idx=0,
    ),
    "watch_history": FeatureSpec(
        type="categorical",
        embed_size=64,
        vocab_size=500,
        sequence_len=50,
        padding_idx=0,
    ),
    "percent_watched": FeatureSpec(
        type="numerical", embed_size=10, sequence_len=50
    ),
}
model = MaskNet(feature_specs)
model.compile()

torch.save(
    model.state_dict(), 
    os.path.join(base_dir, sub_dir, "model.pth")
)

# Exporting feature configs
model.embedding_layer.to_config(
    os.path.join(base_dir, sub_dir, "feature_config.yaml")
)



#%% Save .mar file
# This simply creates a .zip file. You can unzip the folder 
# and see the contents of it using unzip model.mar
export_to_model_archive(
    model_name="masknet_recommender", model_version="1.0.0", 
    model_file=os.path.join(base_dir, sub_dir, "model.py"),
    serialized_file=os.path.join(base_dir, sub_dir, "model.pth"),
    handler_file=os.path.join(base_dir, sub_dir, "handler:ModelHandler"),
    export_path=os.path.join(base_dir, sub_dir),
    extra_files=[os.path.join(base_dir, sub_dir, "feature_config.yaml")],
    overwrite=True,
)

# %% Start the torchserve service
