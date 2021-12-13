"""
I, Elliot Hicks, have read and understood the School's Academic Integrity
Policy, as well as guidance relating to this module, and confirm that this
submission complies with the policy. The content of this file is my own
original work, with any significant material copied or adapted from other
sources clearly indicated and attributed.
Author: Elliot Hicks
Project Title: RL_CNN_maze_solver
Date: 13/12/2021
"""


import numpy as np
import torch
import torch.nn as nn
from torch import flatten


class ECNN10(torch.nn.Module):
    def __init__(self, channels, num_actions):
        """

        Parameters
        ----------
        channels : int
            number of channels to expect for input image, for this application,
            channels = 1.
        num_actions : int
            number of actions it can choose from: up, down, left, right

        Returns
        -------
        None. Max pool layers were not used to maintain resolution:
        """
        super(ECNN10, self).__init__()
        self.convolutional_1 = nn.Conv2d(1, 1, kernel_size=2, padding=1)
        self.ReLU_1 = nn.ReLU()
        self.convolutional_2 = nn.Conv2d(1, 1, kernel_size=2, padding=1)
        self.ReLU_2 = nn.ReLU()
        self.fully_connected_in = nn.Linear(in_features=196, out_features=500)
        self.ReLU_3 = nn.ReLU()
        self.fully_connected_out = nn.Linear(
            in_features=500, out_features=num_actions)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        """
        Parameters
        ----------
        x : NumPy ndarray

        Returns
        -------
        x : Torch tensor
            tensor of 4 values, corresponding to action probabilities

        """
        x = (x - np.mean(x)) / np.std(x)
        x = torch.from_numpy(x).float()
        x = self.convolutional_1(x[None, None, ...])
        x = self.ReLU_1(x)
        x = self.convolutional_2(x)
        x = self.ReLU_2(x)
        x = flatten(x, 1)
        x = self.ReLU_3(x)
        x = self.fully_connected_in(x)
        x = self.ReLU_3(x)
        x = self.fully_connected_out(x)
        x = self.softmax(x)

        return x
