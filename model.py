import torch
import torch.nn as nn
import torch.nn.functional as f
import numpy as np
from typing import List


def hidden_init(layer):
    """ see https://arxiv.org/abs/1509.02971 Section 7 for details:
     (CONTINUOUS CONTROL WITH DEEP REINFORCEMENT LEARNING)
    """
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return -lim, lim


class Actor(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size: int, action_size: int, hidden_layer_sizes: List[int], activation_func=f.relu):
        """Initialize parameters and build model.

            :param state_size: Dimension of each state
            :param action_size: Dimension of each action
            :param hidden_layer_sizes: Number of nodes in hidden layers
            :param activation_func: Activation function
        """
        super(Actor, self).__init__()
        self.action_size = action_size
        self.input_norm = nn.BatchNorm1d(state_size)
        self.activation_func = activation_func
        self.input_layer = nn.Linear(state_size, hidden_layer_sizes[0])
        self.hidden_layers = nn.ModuleList()
        self.hidden_input_norms = nn.ModuleList()
        for i in range(len(hidden_layer_sizes) - 1):
            hidden_layer = nn.Linear(hidden_layer_sizes[i], hidden_layer_sizes[i + 1])
            self.hidden_layers.append(hidden_layer)
            self.hidden_input_norms.append(nn.BatchNorm1d(hidden_layer_sizes[i]))
        self.hidden_input_norms.append(nn.BatchNorm1d(hidden_layer_sizes[-1]))
        self.output_layer = nn.Linear(hidden_layer_sizes[-1], action_size)
        self.reset_parameters()

    def reset_parameters(self):
        self.input_layer.weight.data.uniform_(*hidden_init(self.input_layer))
        for hidden_layer in self.hidden_layers:
            hidden_layer.weight.data.uniform_(*hidden_init(hidden_layer))
        self.output_layer.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state: torch.Tensor):
        """Build an actor (policy) network that maps states -> actions.
        Note: Do not call this function directly. Instead, use: actor(state)
        """
        x = self.input_norm(state)
        x = self.activation_func(self.input_layer(x))
        for i, hidden_layer in enumerate(self.hidden_layers):
            x = self.hidden_input_norms[i](x)
            x = self.activation_func(hidden_layer(x))
        x = self.hidden_input_norms[-1](x)
        # this outputs action values in the range -1 to 1 :
        return torch.tanh(self.output_layer(x))

    def __call__(self, state: torch.Tensor) -> torch.Tensor:
        return super().__call__(state)


class Critic(nn.Module):
    """Critic (Value) Model."""

    def __init__(self, state_size, action_size, hidden_layer_sizes: List[int], activation_func=f.relu, inject_layer=0):
        """Initialize parameters and build model.

            :param state_size: Dimension of each state
            :param action_size: Dimension of each action
            :param hidden_layer_sizes: Number of nodes in hidden layers
            :param activation_func: Activation function
            :param inject_layer: The number of the hidden layer to inject action values into
        """
        super(Critic, self).__init__()
        if inject_layer < 0 or inject_layer >= len(hidden_layer_sizes) - 1:
            raise ValueError()
        self.inject_layer = inject_layer
        self.input_norm = nn.BatchNorm1d(state_size)
        self.activation_func = activation_func
        self.input_layer = nn.Linear(state_size, hidden_layer_sizes[0])
        self.hidden_layers = nn.ModuleList()
        self.hidden_input_norms = nn.ModuleList()
        for i in range(len(hidden_layer_sizes) - 1):
            in_features = hidden_layer_sizes[i]
            # insert the action parameters in hidden layer:
            if i == inject_layer:
                in_features += action_size
            hidden_layer = nn.Linear(in_features, hidden_layer_sizes[i + 1])
            self.hidden_layers.append(hidden_layer)
            self.hidden_input_norms.append(nn.BatchNorm1d(hidden_layer_sizes[i]))
        # There's only one Q-value as output, because the input is a state-action pair now (compared to DQN):
        self.output_layer = nn.Linear(hidden_layer_sizes[-1], 1)
        self.reset_parameters()

    def reset_parameters(self):
        self.input_layer.weight.data.uniform_(*hidden_init(self.input_layer))
        for hidden_layer in self.hidden_layers:
            hidden_layer.weight.data.uniform_(*hidden_init(hidden_layer))
        self.output_layer.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state, action):
        """Build a critic (value) network that maps (state, action) pairs -> Q-values.
        Note: Do not call this function directly. Instead, use: critic(state, action)
        """
        x = self.input_norm(state)
        x = self.activation_func(self.input_layer(x))
        for i, hidden_layer in enumerate(self.hidden_layers):
            x = self.hidden_input_norms[i](x)
            # insert the action parameters in hidden layer:
            if i == self.inject_layer:
                x = torch.cat((x, action), dim=1)
            x = self.activation_func(hidden_layer(x))
        return self.output_layer(x)

    def __call__(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return super().__call__(state, action)
