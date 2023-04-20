import math
from functools import reduce
from operator import __add__
from typing import Optional

import numpy as np
import torch
import torch.nn as nn


def build_linear_positions(index_dims, output_range=(-1.0, 1.0)):
    """
    Generate an array of position indices for an N-D input array.
    Args:
      index_dims (`List[int]`):
        The shape of the index dimensions of the input array.
      output_range (`Tuple[float]`, *optional*, defaults to `(-1.0, 1.0)`):
        The min and max values taken by each input index dimension.
    Returns:
      `torch.FloatTensor` of shape `(index_dims[0], index_dims[1], .., index_dims[-1], N)`.
    """

    def _linspace(n_xels_per_dim):
        return torch.linspace(start=output_range[0], end=output_range[1], steps=n_xels_per_dim, dtype=torch.float32)

    dim_ranges = [_linspace(n_xels_per_dim) for n_xels_per_dim in index_dims]
    array_index_grid = torch.meshgrid(*dim_ranges)

    return torch.stack(array_index_grid, dim=-1)


def _check_or_build_spatial_positions(pos, index_dims, batch_size):
    """
    Checks or builds spatial position features (x, y, ...).
    Args:
      pos (`torch.FloatTensor`):
        None, or an array of position features. If None, position features are built. Otherwise, their size is checked.
      index_dims (`List[int]`):
        An iterable giving the spatial/index size of the data to be featurized.
      batch_size (`int`):
        The batch size of the data to be featurized.
    Returns:
        `torch.FloatTensor` of shape `(batch_size, prod(index_dims))` an array of position features.
    """
    if pos is None:
        pos = build_linear_positions(index_dims)
        pos = torch.broadcast_to(pos[None], (batch_size,) + pos.shape)
        pos = torch.reshape(pos, [batch_size, np.prod(index_dims), -1])

    else:
        # Just a warning label: you probably don't want your spatial features to
        # have a different spatial layout than your pos coordinate system.
        # But feel free to override if you think it'll work!
        if pos.shape[-1] != len(index_dims):
            raise ValueError("Spatial features have the wrong number of dimensions.")
    return pos


def generate_fourier_features(pos, num_bands, max_resolution=(224, 224), concat_pos=True, sine_only=False):
    """
    Generate a Fourier frequency position encoding with linear spacing.
    Args:
      pos (`torch.LongTensor` of shape `(batch_size, sequence_length, dim)`):
        The Tensor containing the position of n points in d dimensional space.
      num_bands (`int`):
        The number of frequency bands (K) to use.
      max_resolution (`Tuple[int]`, *optional*, defaults to (224, 224)):
        The maximum resolution (i.e. the number of pixels per dim). A tuple representing resolution for each dimension.
      concat_pos (`bool`, *optional*, defaults to `True`):
        Whether to concatenate the input position encoding to the Fourier features.
      sine_only (`bool`, *optional*, defaults to `False`):
        Whether to use a single phase (sin) or two (sin/cos) for each frequency band.
    Returns:
      `torch.FloatTensor` of shape `(batch_size, sequence_length, n_channels)`: The Fourier position embeddings. If
      `concat_pos` is `True` and `sine_only` is `False`, output dimensions are ordered as: [dim_1, dim_2, ..., dim_d,
      sin(pi*f_1*dim_1), ..., sin(pi*f_K*dim_1), ..., sin(pi*f_1*dim_d), ..., sin(pi*f_K*dim_d), cos(pi*f_1*dim_1),
      ..., cos(pi*f_K*dim_1), ..., cos(pi*f_1*dim_d), ..., cos(pi*f_K*dim_d)], where dim_i is pos[:, i] and f_k is the
      kth frequency band.
    """

    batch_size = pos.shape[0]

    min_freq = 1.0
    # Nyquist frequency at the target resolution:
    freq_bands = torch.stack(
        [torch.linspace(start=min_freq, end=res / 2, steps=num_bands) for res in max_resolution], dim=0
    )

    # Get frequency bands for each spatial dimension.
    # Output is size [n, d * num_bands]
    per_pos_features = pos[0, :, :][:, :, None] * freq_bands[None, :, :]
    per_pos_features = torch.reshape(per_pos_features, [-1, np.prod(per_pos_features.shape[1:])])

    if sine_only:
        # Output is size [n, d * num_bands]
        per_pos_features = torch.sin(np.pi * (per_pos_features))
    else:
        # Output is size [n, 2 * d * num_bands]
        per_pos_features = torch.cat(
            [torch.sin(np.pi * per_pos_features), torch.cos(np.pi * per_pos_features)], dim=-1
        )

    # Concatenate the raw input positions.
    if concat_pos:
        # Adds d bands to the encoding.
        per_pos_features = torch.cat([pos, per_pos_features.expand(batch_size, -1, -1)], dim=-1)

    return per_pos_features


class PerceiverFourierPositionEncoding(nn.Module):
    """Fourier (Sinusoidal) position encoding."""

    def __init__(self, num_bands, max_resolution, concat_pos=True, sine_only=False):
        super().__init__()
        self.num_bands = num_bands
        self.max_resolution = max_resolution
        self.concat_pos = concat_pos
        self.sine_only = sine_only

    @property
    def num_dimensions(self) -> int:
        return len(self.max_resolution)

    def output_size(self):
        """Returns size of positional encodings last dimension."""
        num_dims = len(self.max_resolution)
        encoding_size = self.num_bands * num_dims
        if not self.sine_only:
            encoding_size *= 2
        if self.concat_pos:
            encoding_size += self.num_dimensions

        return encoding_size

    def forward(self, index_dims, batch_size, device, pos=None):
        pos = _check_or_build_spatial_positions(pos, index_dims, batch_size)
        fourier_pos_enc = generate_fourier_features(
            pos,
            num_bands=self.num_bands,
            max_resolution=self.max_resolution,
            concat_pos=self.concat_pos,
            sine_only=self.sine_only,
        ).to(device)
        return fourier_pos_enc


class PerceiverTrainablePositionEncoding(nn.Module):
    """Trainable position encoding."""

    def __init__(self, index_dims, num_channels=128):
        super().__init__()
        self._num_channels = num_channels
        self._index_dims = index_dims
        index_dim = np.prod(index_dims)
        self.position_embeddings = nn.Parameter(torch.randn(index_dim, num_channels))

    @property
    def num_dimensions(self) -> int:
        if isinstance(self._index_dims, int):
            return 1
        return len(self._index_dims)

    def output_size(self, *args, **kwargs) -> int:
        return self._num_channels

    def forward(self, batch_size):
        position_embeddings = self.position_embeddings

        if batch_size is not None:
            position_embeddings = position_embeddings.expand(batch_size, -1, -1)
        return position_embeddings


def build_position_encoding(
    position_encoding_type,
    out_channels=None,
    project_pos_dim=-1,
    trainable_position_encoding_kwargs=None,
    fourier_position_encoding_kwargs=None,
):
    """
    Builds the position encoding.
    Args:
    - out_channels: refers to the number of channels of the position encodings.
    - project_pos_dim: if specified, will project the position encodings to this dimension.
    """

    if position_encoding_type == "trainable":
        if not trainable_position_encoding_kwargs:
            raise ValueError("Make sure to pass trainable_position_encoding_kwargs")
        output_pos_enc = PerceiverTrainablePositionEncoding(**trainable_position_encoding_kwargs)
    elif position_encoding_type == "fourier":
        # We don't use the index_dims argument, as this is only known during the forward pass
        if not fourier_position_encoding_kwargs:
            raise ValueError("Make sure to pass fourier_position_encoding_kwargs")
        output_pos_enc = PerceiverFourierPositionEncoding(**fourier_position_encoding_kwargs)
    else:
        raise ValueError(f"Unknown position encoding type: {position_encoding_type}.")

    # Optionally, project the position encoding to a target dimension:
    positions_projection = nn.Linear(out_channels, project_pos_dim) if project_pos_dim > 0 else nn.Identity()

    return output_pos_enc, positions_projection


class Conv2dSamePadding(nn.Conv2d):
    """
    Conv2d layer with padding="same" support. Source:
    https://gist.github.com/sumanmichael/4de9dee93f972d47c80c4ade8e149ea6
    """

    def __init__(self, *args, **kwargs):
        super(Conv2dSamePadding, self).__init__(*args, **kwargs)
        self.zero_pad_2d = nn.ZeroPad2d(
            reduce(__add__, [(k // 2 + (k - 2 * (k // 2)) - 1, k // 2) for k in self.kernel_size[::-1]])
        )

    def forward(self, input):
        return self._conv_forward(self.zero_pad_2d(input), self.weight, self.bias)


class Conv2DDownsample(nn.Module):
    """Downsamples 4x by applying a 2D convolution and doing max pooling."""

    def __init__(
        self,
        num_layers: int = 1,
        in_channels: int = 3,
        out_channels: int = 64,
        use_batchnorm: bool = True,
    ):
        """
        Constructs a Conv2DDownsample model.
        Args:
          in_channels (`int`, *optional*, defaults to 3):
            The number of input channels.
          out_channels (`int`, *optional*, defaults to 64):
            The number of conv output channels.
          use_batchnorm (`bool`, *optional*, defaults to `True`):
            Whether to use batchnorm.
        """
        super().__init__()

        self.conv = Conv2dSamePadding(
            in_channels=in_channels, out_channels=out_channels, kernel_size=7, stride=2, bias=False
        )
        self.batchnorm = nn.BatchNorm2d(num_features=out_channels) if use_batchnorm else nn.Identity()
        self.relu = nn.ReLU()
        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        out = self.conv(inputs)
        out = self.batchnorm(out)
        out = self.relu(out)
        out = self.max_pool(out)
        return out


def space_to_depth(frames: torch.Tensor, temporal_block_size: int = 1, spatial_block_size: int = 1) -> torch.Tensor:
    """
    Space to depth transform. Rearranges blocks of spatial data, into depth.
    This function assumes the channels to be first, but will place the channels last after transformation.
    Based on https://discuss.pytorch.org/t/is-there-any-layer-like-tensorflows-space-to-depth-function/3487/15.
    """
    if len(frames.shape) == 4:
        batch_size, num_channels, height, width = frames.shape
        # split up dimensions (height by spatial_block_size, width by spatial_block_size)
        frames = frames.view(
            batch_size,
            num_channels,
            height // spatial_block_size,
            spatial_block_size,
            width // spatial_block_size,
            spatial_block_size,
        )
        # move blocks to last dimension: (batch_size, H//bs, W//bs, bs, bs, C)
        frames = frames.permute(0, 2, 4, 3, 5, 1).contiguous()
        # concatenate blocks along channel dimension: (batch_size, H//bs, W//bs, bs*bs*C)
        frames = frames.view(
            batch_size,
            height // spatial_block_size,
            width // spatial_block_size,
            (spatial_block_size**2) * num_channels,
        )
        return frames
    elif len(frames.shape) == 5:
        batch_size, time, num_channels, height, width = frames.shape
        # split up dimensions (time by temporal_block_size, height by spatial_block_size, width by spatial_block_size)
        frames = frames.view(
            batch_size,
            time // temporal_block_size,
            temporal_block_size,
            num_channels,
            height // spatial_block_size,
            spatial_block_size,
            width // spatial_block_size,
            spatial_block_size,
        )
        # move blocks to last dimension: (batch_size, T//ts, H//bs, W//bs, ts, bs, bs, C)
        frames = frames.permute(0, 1, 4, 6, 2, 5, 7, 3).contiguous()
        # concatenate blocks along channel dimension: (batch_size, T//ts, H//bs, W//bs, ts*bs*bs*C)
        frames = frames.view(
            batch_size,
            time // temporal_block_size,
            height // spatial_block_size,
            width // spatial_block_size,
            temporal_block_size * (spatial_block_size**2) * num_channels,
        )
        return frames
    else:
        raise ValueError(
            "Frames should be of rank 4 (batch, channels, height, width)"
            " or rank 5 (batch, time, channels, height, width)"
        )


class PerceiverImagePreprocessor(nn.Module):
    """
    Image preprocessing for Perceiver Encoder.
    Note: the *out_channels* argument refers to the output channels of a convolutional layer, if *prep_type* is set to
    "conv1x1" or "conv". If one adds absolute position embeddings, one must make sure the *num_channels* of the
    position encoding kwargs are set equal to the *out_channels*.
    Args:
        cfg ([*PerceiverConfig*]):
            Model cfguration.
        prep_type (`str`, *optional*, defaults to `"conv"`):
            Preprocessing type. Can be "conv1x1", "conv", "patches", "pixels".
        spatial_downsample (`int`, *optional*, defaults to 4):
            Spatial downsampling factor.
        temporal_downsample (`int`, *optional*, defaults to 1):
            Temporal downsampling factor (only relevant in case a time dimension is present).
        position_encoding_type (`str`, *optional*, defaults to `"fourier"`):
            Position encoding type. Can be "fourier" or "trainable".
        in_channels (`int`, *optional*, defaults to 3):
            Number of channels in the input.
        out_channels (`int`, *optional*, defaults to 64):
            Number of channels in the output.
        conv_after_patching (`bool`, *optional*, defaults to `False`):
            Whether to apply a convolutional layer after patching.
        conv_after_patching_in_channels (`int`, *optional*, defaults to 54):
            Number of channels in the input of the convolutional layer after patching.
        conv2d_use_batchnorm (`bool`, *optional*, defaults to `True`):
            Whether to use batch normalization in the convolutional layer.
        concat_or_add_pos (`str`, *optional*, defaults to `"concat"`):
            How to concatenate the position encoding to the input. Can be "concat" or "add".
        project_pos_dim (`int`, *optional*, defaults to -1):
            Dimension of the position encoding to project to. If -1, no projection is applied.
        **position_encoding_kwargs (`Dict`, *optional*):
            Keyword arguments for the position encoding.
    """

    def __init__(
        self,
        cfg,
        prep_type="conv",
        spatial_downsample: int = 4,
        temporal_downsample: int = 1,
        position_encoding_type: str = "fourier",
        in_channels: int = 3,
        out_channels: int = 64,
        conv_after_patching: bool = False,
        conv_after_patching_in_channels: int = 99,  # only relevant when conv_after_patching = True
        conv2d_use_batchnorm: bool = True,
        concat_or_add_pos: str = "concat",
        project_pos_dim: int = -1,
        **position_encoding_kwargs,
    ):
        super().__init__()
        self.cfg = cfg

        if prep_type not in ("conv", "patches", "pixels", "conv1x1"):
            raise ValueError(f"Prep_type {prep_type} is invalid")

        if concat_or_add_pos not in ["concat", "add"]:
            raise ValueError(f"Invalid value {concat_or_add_pos} for concat_or_add_pos.")

        self.in_channels = in_channels
        self.prep_type = prep_type
        self.spatial_downsample = spatial_downsample
        self.temporal_downsample = temporal_downsample
        self.position_encoding_type = position_encoding_type
        self.concat_or_add_pos = concat_or_add_pos
        self.conv_after_patching = conv_after_patching
        self.out_channels = out_channels

        if self.prep_type == "conv":
            # Downsampling with conv is currently restricted
            convnet_num_layers = math.log(spatial_downsample, 4)
            convnet_num_layers_is_int = convnet_num_layers == np.round(convnet_num_layers)
            if not convnet_num_layers_is_int or temporal_downsample != 1:
                raise ValueError(
                    "Only powers of 4 expected for spatial and 1 expected for temporal downsampling with conv."
                )
            self.convnet = Conv2DDownsample(
                in_channels=in_channels,
                num_layers=int(convnet_num_layers),
                out_channels=out_channels,
                use_batchnorm=conv2d_use_batchnorm,
            )

        elif self.prep_type == "conv1x1":
            if temporal_downsample != 1:
                raise ValueError("Conv1x1 does not downsample in time.")
            self.convnet_1x1 = nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=(1, 1),
                # spatial_downsample is unconstrained for 1x1 convolutions.
                stride=(spatial_downsample, spatial_downsample),
            )

        # Position embeddings
        self.project_pos_dim = project_pos_dim
        self.position_embeddings, self.positions_projection = build_position_encoding(
            position_encoding_type=position_encoding_type,
            out_channels=out_channels,
            project_pos_dim=project_pos_dim,
            **position_encoding_kwargs,
        )

        # Optional convolutional layer after patches.
        self.conv_after_patches = (
            nn.Linear(conv_after_patching_in_channels, self.out_channels) if conv_after_patching else nn.Identity()
        )

    @property
    def num_channels(self) -> int:
        # Let's assume that the number of resolutions (in the context of image preprocessing)
        # of the input data is 2 or 3 depending on whether we are processing image or video respectively.
        # In this case, for convenience, we will declare is_temporal variable,
        # which will show whether the data has a temporal dimension or not.
        is_temporal = self.position_embeddings.num_dimensions > 2

        # position embedding
        if self.project_pos_dim > 0:
            pos_dim = self.project_pos_dim
        else:
            pos_dim = self.position_embeddings.output_size()
        if self.concat_or_add_pos == "add":
            return pos_dim

        # inputs
        if self.conv_after_patching or self.prep_type in ("conv1x1", "conv"):
            inp_dim = self.out_channels
        elif self.prep_type == "pixels":
            inp_dim = self.in_channels
            if not is_temporal:
                inp_dim = math.ceil(inp_dim / self.spatial_downsample)
        elif self.prep_type == "patches":
            if self.conv_after_patching:
                inp_dim = self.out_channels
            else:
                inp_dim = self.in_channels * self.spatial_downsample**2
                if is_temporal:
                    inp_dim *= self.temporal_downsample

        return inp_dim + pos_dim

    def _build_network_inputs(self, inputs: torch.Tensor, pos: torch.Tensor, network_input_is_1d: bool = True):
        """
        Construct the final input, including position encoding.
        This method expects the inputs to always have channels as last dimension.
        """
        batch_size = inputs.shape[0]
        index_dims = inputs.shape[1:-1]
        indices = np.prod(index_dims)

        # Flatten input features to a 1D index dimension if necessary.
        if len(inputs.shape) > 3 and network_input_is_1d:
            inputs = torch.reshape(inputs, [batch_size, indices, -1])

        # Construct the position encoding.
        if self.position_encoding_type == "trainable":
            pos_enc = self.position_embeddings(batch_size)
        elif self.position_encoding_type == "fourier":
            pos_enc = self.position_embeddings(index_dims, batch_size, device=inputs.device)

        # Optionally project them to a target dimension.
        pos_enc = self.positions_projection(pos_enc)

        if not network_input_is_1d:
            # Reshape pos to match the input feature shape
            # if the network takes non-1D inputs
            sh = inputs.shape
            pos_enc = torch.reshape(pos_enc, list(sh)[:-1] + [-1])
        if self.concat_or_add_pos == "concat":
            inputs_with_pos = torch.cat([inputs, pos_enc], dim=-1)
        elif self.concat_or_add_pos == "add":
            inputs_with_pos = inputs + pos_enc
        return inputs_with_pos, inputs

    def forward(self, inputs: torch.Tensor, pos: Optional[torch.Tensor] = None, network_input_is_1d: bool = True):
        if self.prep_type == "conv":
            # Convnet image featurization.
            # Downsamples spatially by a factor of 4
            inputs = self.convnet(inputs)

        elif self.prep_type == "conv1x1":
            # map inputs to self.out_channels
            inputs = self.convnet_1x1(inputs)

        elif self.prep_type == "pixels":
            # if requested, downsamples in the crudest way
            if inputs.ndim == 4:
                inputs = inputs[:: self.spatial_downsample, :: self.spatial_downsample]
            elif inputs.ndim == 5:
                inputs = inputs[
                    :, :: self.temporal_downsample, :, :: self.spatial_downsample, :: self.spatial_downsample
                ]
            else:
                raise ValueError("Unsupported data format for pixels.")

        elif self.prep_type == "patches":
            # Space2depth featurization.
            # Video: B x T x C x H x W
            inputs = space_to_depth(
                inputs, temporal_block_size=self.temporal_downsample, spatial_block_size=self.spatial_downsample
            )

            if inputs.ndim == 5 and inputs.shape[1] == 1:
                # for flow
                inputs = inputs.squeeze(dim=1)

            # Optionally apply conv layer.
            inputs = self.conv_after_patches(inputs)

        if self.prep_type != "patches":
            # move channels to last dimension, as the _build_network_inputs method below expects this
            if inputs.ndim == 4:
                inputs = torch.moveaxis(inputs, 1, -1)
            elif inputs.ndim == 5:
                inputs = torch.moveaxis(inputs, 2, -1)
            else:
                raise ValueError("Unsupported data format for conv1x1.")

        inputs, inputs_without_pos = self._build_network_inputs(inputs, pos, network_input_is_1d)
        modality_sizes = None  # Size for each modality, only needed for multimodal

        return inputs, modality_sizes, inputs_without_pos
