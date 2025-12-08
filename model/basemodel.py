import torch
from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple, Union, List

import torch.nn as nn

class BaseDetectionModel(nn.Module, ABC):
    """Base class for detection models"""

    def __init__(self):
        super(BaseDetectionModel, self).__init__()

    @abstractmethod
    def forward(self, inputs: Dict[str, Any], targets: Dict[str, Any] = None) -> Union[Dict[str, torch.Tensor], torch.Tensor]:
        """
        Forward pass of the model

        Args:
            inputs: Dictionary containing input tensors and metadata
            targets: Dictionary containing target tensors (provided during training, None during inference)

        Returns:
            During training (targets is not None): Dictionary of loss values
            During inference (targets is None): Dictionary of predictions
        """
        pass

    def train_step(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """
        Perform a single training step

        Args:
            batch: Dictionary containing inputs and targets

        Returns:
            Dictionary of loss values
        """
        inputs = {k: v for k, v in batch.items() if k != 'targets'}
        targets = batch.get('targets', None)

        return self.forward(inputs, targets)

    @torch.no_grad()
    def test_step(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform a single test/inference step

        Args:
            batch: Dictionary containing inputs

        Returns:
            Dictionary of predictions
        """
        inputs = {k: v for k, v in batch.items() if k != 'targets'}

        return self.forward(inputs, None)