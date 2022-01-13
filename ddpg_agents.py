from ddpg_agent import DDPGAgent
from utilities import convert_to_numpy
import torch
import numpy as np
from typing import List, Tuple


class DDPGAgents:
    def __init__(self, ddpg_agents: List[DDPGAgent]):
        self.ddpg_agents = ddpg_agents
        self.num_agents = len(ddpg_agents)

    def act(self, agent_states: torch.Tensor, noise_scale: float) -> np.ndarray:
        """ Get actions from all agents

        :param agent_states: states for each agent -> tensor[num_agents, batch_size, state_size]
        :param noise_scale: the amount of noise to add to action values
        :return: np.ndarray[num_agents, batch_size, action_size]
        """
        actions = []
        for i, ddpg_agent in enumerate(self.ddpg_agents):
            states = agent_states[i]
            noise = np.random.normal(scale=noise_scale, size=ddpg_agent.action_size)
            action = convert_to_numpy(ddpg_agent.act(states)) + noise
            actions.append(action)
        return np.stack(actions)

    def step(self, samples: List[Tuple[torch.Tensor, ...]]):
        """
        :param samples: list[num_agents] of tuple(states, actions, rewards, next_states, dones).
            Each element in the tuple is a tensor[num_samples, num_agents, *]
        """
        for i, ddpg_agent, samples_for_agent in zip(range(len(self)), self.ddpg_agents, samples):
            # transpose samples_for_agent to tuple of tensor[num_agents, num_samples, *]:
            samples_for_agent = tuple(torch.transpose(t, 0, 1) for t in samples_for_agent)
            # convert samples_for_agent to tuple of tensor[num_samples, *]:
            samples_for_agent = tuple(t[i] for t in samples_for_agent)
            ddpg_agent.step(samples_for_agent)

    def update_target_networks(self):
        for ddpg_agent in self.ddpg_agents:
            ddpg_agent.update_target_networks()

    def save_checkpoint(self, filename: str):
        state_dicts_list = []
        for ddpg_agent in self.ddpg_agents:
            state_dicts = ddpg_agent.get_state_dicts()
            state_dicts_list.append(state_dicts)
        torch.save(state_dicts_list, filename)

    def load_checkpoint(self, filename):
        state_dicts_list = torch.load(filename)
        for ddpg_agent, state_dicts in zip(self.ddpg_agents, state_dicts_list):
            ddpg_agent.load_state_dicts(state_dicts)

    def __len__(self):
        """Return number of agents."""
        return self.num_agents
#
# class GaussianNoise:
#     def sample(self, output_shape, noise_scale):
#         return np.random.normal(scale=noise_scale, size=output_shape)
