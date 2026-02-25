from abc import abstractmethod
from functools import reduce
import torch.nn as nn
from torchmetrics import MetricCollection
from typing import Union

from common.constants.model_stages import ModelStage

class ModelMixin:
    def _add_chained_blocks(self, 
                            target: nn.Sequential,
                            channel_sizes: Union[list[int],tuple[int]], 
                            name_prefix: str, 
                            block_class: type) -> nn.Sequential:
        """
        Adds a chain of layers to a Sequential container, automatically wiring
        each block's output channels as the next block's input channels.

        Args:
            target: The nn.Sequential container to attach layers to.
            channel_sizes: List or tuple of channel dimensions. Consecutive pairs become
                          (in_channels, out_channels) for each block.
                          e.g. [1, 32, 64, 128] → Block(1,32), Block(32,64), Block(64,128)
            name_prefix: Naming prefix for each layer, suffixed with a 1-based index.
                          e.g. "conv_block_" → "conv_block_1", "conv_block_2", ...
            block_class: The layer constructor to instantiate. Must accept (in_channels, out_channels).
                          e.g. Conv2DBlock, DenseBlock

        Returns:
            The same target Sequential with all blocks attached.
        """
        named_blocks = [
            (f'{name_prefix}{i}', block_class(in_ch, out_ch))
            for i, (in_ch, out_ch) in enumerate(zip(channel_sizes, channel_sizes[1:]), start=1)
        ]

        return reduce(
            lambda seq, layer: (seq.add_module(layer[0], layer[1]), seq)[1],
            named_blocks,
            target,
        )
    def _add_multiple_layers(self,target: nn.Sequential,
                             layers:Union[list[tuple[str,nn.Module]],tuple[tuple[str,nn.Module]]]):
        """
        Adds multiple layers at once, mainly used for readability

        Args:
            target: The nn.Sequential container to attach layers to.
            layers: a list or a tuple of tuples containing the layer name and the nn module of the layer

        Returns:
            The same target Sequential with all blocks attached.
        """
        for name,layer in layers:
            target.add_module(name=name,
                              module=layer)
        return target

    @abstractmethod
    def _default_metrics(self) -> MetricCollection:
        """
        Return the default MetricCollection for this model.
        Each subclass must provide its own task-specific defaults.
        """
        ...

    def _init_metrics(self, metrics) -> MetricCollection:
        """
        Accept user-provided metrics or fall back to subclass defaults.
        """
        if metrics is not None:
            return MetricCollection(metrics) if isinstance(metrics, dict) else metrics
        return self._default_metrics()

    def _setup_metrics(self, metrics):
        """
        Clone the base MetricCollection for each model stage (train/val/test)
        and store them in an nn.ModuleDict keyed by stage prefix.
        """
        base_metrics = self._init_metrics(metrics)

        self.model_stage_metrics = nn.ModuleDict({
            ModelStage.TRAIN.prefix: base_metrics.clone(prefix=ModelStage.TRAIN.prefix),
            ModelStage.VAL.prefix: base_metrics.clone(prefix=ModelStage.VAL.prefix),
            ModelStage.TEST.prefix: base_metrics.clone(prefix=ModelStage.TEST.prefix),
        })