from functools import reduce
import torch.nn as nn

class ModelMixin:
    def _add_chained_blocks(self, target: nn.Sequential, channel_sizes: list[int], name_prefix: str, block_class: type) -> nn.Sequential:
        """
        Adds a chain of layers to a Sequential container, automatically wiring
        each block's output channels as the next block's input channels.

        Args:
            target: The nn.Sequential container to attach layers to.
            channel_sizes: List of channel dimensions. Consecutive pairs become
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
    def _add_multiple_layers(self,target: nn.Sequential,layers:list[tuple[str,nn.Module]]):
        """
        Adds multiple layers at once, mainly used for readability

        Args:
            target: The nn.Sequential container to attach layers to.
            layers: a list of tuples containing the layer name and the nn module of the layer

        Returns:
            The same target Sequential with all blocks attached.
        """
        for name,layer in layers:
            target.add_module(name=name,
                              module=layer)
        return target