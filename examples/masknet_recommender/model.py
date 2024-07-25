from typing import Dict, Literal, List, Optional, TypeVar, Union
from dataclasses import dataclass, asdict
import yaml
import torch
from torch import nn
from torch.nn import functional as F

from pdb import set_trace


@dataclass
class FeatureSpec:
    type: Literal["categorical", "numerical"]
    embed_size: int
    vocab_size: int = None  # only valid for categorical features
    padding_idx: int = None  # only valid for categorical features
    sequence_len: int = (
        None  # non-null values indicate sequential features
    )


class FeatureEmbedding(nn.Module):
    def __init__(self, feature_specs: Dict[str, FeatureSpec]):
        super().__init__()
        self.feature_specs = feature_specs
        # Create embedding layers
        self.embeddings = {}
        self.embed_dim = 0
        for feat, spec in feature_specs.items():
            if spec.type == "categorical":
                self.embeddings[feat] = nn.Embedding(
                    spec.vocab_size + int(spec.padding_idx is not None),
                    spec.embed_size,
                    padding_idx=spec.padding_idx,
                )
            elif spec.type == "numerical":  # numerical
                self.embeddings[feat] = nn.Linear(
                    spec.sequence_len or 1, spec.embed_size
                )
            self.embed_dim += spec.embed_size

    def forward(self, x):
        # Get embeddings
        v_embed = []
        v_embed_ln = []  # layer_normalized embeddings
        for feat, val in x.items():
            if feat not in self.embeddings:
                continue
            # (batch_size, emb_size)
            emb = self.embeddings[feat](val)

            # Aggregate categorical sequence features
            # In practice, this can insert other sequence models
            # such as transformer to create sequence embedding
            if (
                self.feature_specs[feat].type == "categorical"
                and self.feature_specs[feat].sequence_len is not None
            ):
                emb = torch.mean(emb, dim=1)

            # Aggregate
            v_embed.append(emb)
            embed_dim = self.feature_specs[feat].embed_size
            v_embed_ln.append(F.layer_norm(emb, (embed_dim,)))

        v_embed = torch.cat(v_embed, dim=-1)
        v_embed_ln = torch.cat(v_embed_ln, dim=-1)

        return v_embed, v_embed_ln

    @classmethod
    def from_config(cls, config_file: str):
        with open(config_file, "r") as fid:
            content: Dict = yaml.safe_load(fid)

        feature_specs = {}
        for feat, spec in content.items():
            feature_specs[feat] = FeatureSpec(**spec)
        return cls(feature_specs)

    def to_config(self, output_file: str):
        content = {k: asdict(v) for k, v in self.feature_specs.items()}
        with open(output_file, "w") as fid:
            yaml.safe_dump(content, fid)


class InstanceGuidedMask(nn.Module):
    def __init__(
        self,
        input_dim: int,
        aggregation_dim: int,
        projection_dim: Optional[int] = None,
    ):
        super().__init__()
        self.aggregation_layer = nn.Linear(input_dim, aggregation_dim)
        self.projection_layer = nn.Linear(
            aggregation_dim, projection_dim or input_dim
        )

    def forward(self, x):
        # x is embedding of shape (batch_size, seq_len, emb_size)
        mask = self.aggregation_layer(x)
        mask = F.relu(mask)
        mask = self.projection_layer(mask)
        return mask


class MaskBlock(nn.Module):
    def __init__(
        self,
        input_dim: int,
        mask_hidden_dim: int,
        mask_output_dim: Optional[int] = None,
        output_dim: Optional[int] = None,
    ):
        """
        MaskBlock. Can be used for both MaskBlock on Feature Embedding
        and MaskBlock on MaskBlock

        Parameters
        ----------
        input_dim : int
            Input dimension of the block
        mask_hidden_dim : int
            instance guided mask hidden dimension
        mask_output_dim: int
            instance guided mask output dimension. If not specified,
            returns a mask of the same size as input_dim
        output_dim : int, optional
            optionally specify an output dimension of the block.
            Otherwise, the block will output features of the same size
            as the input
        """
        super().__init__()
        self.instance_guided_mask = InstanceGuidedMask(
            input_dim, mask_hidden_dim, mask_output_dim
        )
        self.hidden = nn.Linear(
            mask_output_dim or input_dim,
            output_dim or input_dim,
            bias=False,
        )
        self.layer_norm = nn.LayerNorm((output_dim or input_dim,))

    def forward(self, x, output):
        mask = self.instance_guided_mask(x)  # masked features

        # Apply mask:
        # output: previous maskblock output,
        # or individually layer normalized embeddings
        y = mask * output
        y = self.hidden(y)

        # Output hidden
        y = self.layer_norm(y)
        y = F.relu(y)

        return y


class MaskNet(nn.Module):
    """
    MaskNet implementation for both serial and parallel architecture.

    Parameters
    ----------
    feature_specs : Dict[str, FeatureSpec]
        Dictionary of feature specs.
    num_blocks : int, optional
        Number of MaskBlocks, by default 3
    block_output_size : int, optional
        Size of output for each block, by default 128
    mask_hidden_dim : int, optional
        Hidden or aggregation layer size of the mask,
        by default 256
    architecture : Literal['serial', 'parallel'], optional
        MaskNet architecture, by default "serial"
    parallel_projection_sizes : List[int], optional
        This is only relevant for parallel model architecture,
        specifying the hidden layer sizes before output,
        by default [128, 128]
    """

    def __init__(
        self,
        feature_specs: Dict[str, FeatureSpec],
        num_blocks: int = 3,
        block_output_size: int = 128,
        mask_hidden_dim: int = 256,
        architecture: Literal["serial", "parallel"] = "parallel",
        parallel_projection_sizes: List[int] = [128, 128],
    ):
        super().__init__()
        self.block_output_size = block_output_size
        self.mask_hidden_dim = mask_hidden_dim
        self.architecture = architecture
        self.parallel_projection_sizes = parallel_projection_sizes

        # Initialize the embedding layers for features
        self.embedding_layer = FeatureEmbedding(feature_specs)

        # Create layers depending on architecture
        if architecture == "parallel":
            self.initialize_parallel_layers(
                num_blocks,
                mask_hidden_dim,
                block_output_size,
                parallel_projection_sizes,
            )
        elif architecture == "serial":
            self.initialize_serial_layers(
                num_blocks, mask_hidden_dim, block_output_size
            )

    def initialize_parallel_layers(
        self,
        num_blocks: int,
        mask_hidden_dim: int,
        block_output_size: int,
        parallel_projection_sizes: List[int],
    ):
        """Create layers for parallel architecture."""
        # Mask blocks
        self.mask_blocks = nn.ModuleList()
        for _ in range(num_blocks):
            block = MaskBlock(
                input_dim=self.embedding_layer.embed_dim,
                mask_hidden_dim=mask_hidden_dim,
                # parallel mask.shape == feature.shape
                mask_output_dim=None,
                output_dim=block_output_size,
            )
            self.mask_blocks.append(block)

        # Output projection layers
        hidden_size = (
            block_output_size * num_blocks
            if block_output_size
            else self.embedding_layer.embed_dim * num_blocks
        )
        parallel_projection_sizes = [
            hidden_size
        ] + parallel_projection_sizes

        self.output_projection = nn.Sequential()
        for ii in range(len(parallel_projection_sizes) - 1):
            f"parallel_output_projection_{ii}"
            self.output_projection.add_module(
                f"parallel_output_projection_{ii}",
                nn.Linear(
                    parallel_projection_sizes[ii],
                    parallel_projection_sizes[ii + 1],
                ),
            )
        # Output projection
        self.prediction_layer = nn.Linear(
            parallel_projection_sizes[-1], 1
        )

    def initialize_serial_layers(
        self,
        num_blocks: int,
        mask_hidden_dim: int,
        block_output_size: int = None,
    ):
        """Create layers for serial architecture."""
        # Mask blocks
        self.mask_blocks = nn.ModuleList()
        for ii in range(num_blocks):
            block = MaskBlock(
                input_dim=self.embedding_layer.embed_dim,
                mask_hidden_dim=mask_hidden_dim,
                # First block is masking on the feature, so the mask
                # size needs to be the same as the input
                # the rest of the layers follow block_output_size
                mask_output_dim=None if ii == 0 else block_output_size,
                output_dim=block_output_size,
            )
            self.mask_blocks.append(block)

        # Output projection
        self.prediction_layer = nn.Linear(
            block_output_size or self.embedding_layer.embed_dim, 1
        )

    def forward(self, x, y=None):
        v_embed, v_embed_ln = self.embedding_layer(x)
        if self.architecture == "serial":
            z = self._forward_serial(v_embed, v_embed_ln)
        elif self.architecture == "parallel":
            z = self._forward_parallel(v_embed, v_embed_ln)
        z = self.prediction_layer(z)

        return z

    def _forward_parallel(self, v_embed, v_embed_ln):
        hidden = []
        for block in self.mask_blocks:
            hidden.append(block(v_embed, v_embed_ln))
        hidden = torch.cat(hidden, dim=-1)
        projection = self.output_projection(hidden)
        return projection

    def _forward_serial(self, v_embed, v_embed_ln):
        mask_block_output = v_embed_ln
        for block in self.mask_blocks:
            mask_block_output = block(v_embed, mask_block_output)
        return mask_block_output

    @classmethod
    def from_config(cls, config_file: str):
        with open(config_file, "r") as fid:
            configs: Dict = yaml.safe_load(fid)
        # Convert feature_specs
        feature_specs = {}
        for feat, spec in configs["feature_specs"].items():
            feature_specs[feat] = FeatureSpec(**spec)
        configs["feature_specs"] = feature_specs
        return cls(**configs)

    def to_config(self, output_file: str):
        # Convert feature specs to dict
        feature_specs = {
            k: asdict(v)
            for k, v in self.embedding_layer.feature_specs.items()
        }
        config = {"feature_specs": feature_specs}
        config["num_blocks"] = len(self.mask_blocks)
        config["block_output_size"] = self.block_output_size
        config["mask_hidden_dim"] = self.mask_hidden_dim
        config["architecture"] = self.architecture
        config["parallel_projection_sizes"] = (
            self.parallel_projection_sizes
        )

        # Write
        with open(output_file, "w") as fid:
            yaml.safe_dump(config, fid)


if __name__ == "__main__":
    # Test the model implementation
    batch_size = 4
    seq_len = 50
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

    x = {
        "day_of_week": torch.randint(1, 8, (batch_size,)),
        "hour_of_day": torch.randint(0, 24, (batch_size,)),
        "account_tenure": torch.randint(1, 6, (batch_size,)),
        "payment_tier": torch.randint(1, 4, (batch_size,)),
        "watch_history": torch.randint(1, 501, (batch_size, seq_len)),
        "percent_watched": torch.rand((batch_size, seq_len)),
    }

    model = MaskNet(
        feature_specs,
        num_blocks=3,
        block_output_size=17,
        architecture="serial",
    )

    model.eval()
    with torch.no_grad():
        output = model(x)
