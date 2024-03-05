import torch
from torch.nn import _reduction as _Reduction
from typing import Optional
from torch import Tensor
import torch.nn.functional as F


class BCEWithLogitsLoss(torch.nn.modules.Module):
    __name__ = "BCEWithLogitsLoss"  # Add a class attribute __name__

    def __init__(
        self,
        weight: Optional[Tensor] = None,
        size_average=None,
        reduce=None,
        reduction: str = "mean",
        pos_weight: Optional[Tensor] = None,
        ignore_channels: Optional[list] = None,
    ) -> None:
        super(BCEWithLogitsLoss, self).__init__()

        self.ignore_channels = ignore_channels
        self.weight = weight
        self.pos_weight = pos_weight
        self.reduction = reduction

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        # Apply the ignore_channels logic to input and target
        if self.ignore_channels is not None:
            input, target = self._take_channels(input, target)

        return F.binary_cross_entropy_with_logits(
            input,
            target,
            self.weight,
            pos_weight=self.pos_weight,
            reduction=self.reduction,
        )

    def _take_channels(self, input: Tensor, target: Tensor):
        # Function to exclude specified channels
        channels_to_include = [
            channel
            for channel in range(input.shape[1])
            if channel not in self.ignore_channels
        ]
        input = torch.index_select(
            input, dim=1, index=torch.tensor(channels_to_include).to(input.device)
        )
        target = torch.index_select(
            target, dim=1, index=torch.tensor(channels_to_include).to(target.device)
        )
        return input, target

    def to(self, *args, **kwargs):
        # Override the to() method to support moving the loss to a specific device
        self.ignore_channels = [int(c) for c in self.ignore_channels]
        return super(BCEWithLogitsLoss, self).to(*args, **kwargs)
