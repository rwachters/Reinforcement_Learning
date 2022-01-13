import numpy as np
from mlagents_envs.base_env import ActionTuple
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel
from typing import Tuple, List, Optional


class MyUnityEnvironment:
    def __init__(self, file_name=None, no_graphics=False, seed=1, worker_id=0):
        """
           :param file_name: The filename of the Unity executable, or None when using the Unity editor
               (press Play to connect).
           :param no_graphics: Whether to use a graphics window or not.
           :param seed: The seed used for a pseudo random number generator.
           :param worker_id: The id of the Unity thread to create. You cannot create threads with the same id.
        """
        self.engine_configuration_channel = EngineConfigurationChannel()
        side_channels = [self.engine_configuration_channel]
        self.env = UnityEnvironment(file_name=file_name, no_graphics=no_graphics, seed=seed, worker_id=worker_id,
                                    side_channels=side_channels)
        self.env.reset()
        self.behavior_names = sorted(self.env.behavior_specs.keys())
        self.behavior_specs = [self.env.behavior_specs[behavior_name] for behavior_name in self.behavior_names]
        self.num_agents_list = []  # number of agents for each behavior
        for behavior_name in self.behavior_names:
            decision_steps, _ = self.env.get_steps(behavior_name)
            self.num_agents_list.append(len(decision_steps))

    def set_timescale(self, time_scale: float):
        """ Set the timescale at which the physics simulation runs.

        :param time_scale: a value of 1.0 means the simulation runs in realtime.
        """
        self.engine_configuration_channel.set_configuration_parameters(time_scale=time_scale)

    def set_display_size(self, width: int, height: int):
        self.engine_configuration_channel.set_configuration_parameters(width=width, height=height)

    def reset(self):
        self.env.reset()

    def get_observations(self, behavior_index: int) -> np.ndarray:
        """ Get observations for behavior.
        Agents can have different behaviors. For example: Two strikers, and a goalie in the soccer example.

        :return: np.ndarray[num_agents, observation_size]
        """
        num_agents = self.num_agents_list[behavior_index]
        behavior_spec = self.behavior_specs[behavior_index]
        behavior_name = self.behavior_names[behavior_index]
        observations = np.ndarray((num_agents, *behavior_spec.observation_specs[0].shape))
        decision_steps, terminal_steps = self.env.get_steps(behavior_name)
        for agent_id in decision_steps:
            observations[agent_id] = decision_steps[agent_id].obs[0]
        return observations

    def set_actions(self, behavior_index: int, continuous: Optional[np.ndarray] = None,
                    discrete: Optional[np.ndarray] = None):
        """ Set actions for behavior.

        :param behavior_index:
        :param continuous: ndarray[num_agents, *]
        :param discrete:
        """

        behavior_name = self.behavior_names[behavior_index]
        action_tuple = ActionTuple(continuous=continuous, discrete=discrete)
        self.env.set_actions(behavior_name, action_tuple)

    def step(self):
        """ Step forward in environment. """
        self.env.step()

    def get_experiences(self, behavior_index: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """ Get experiences for all agents with behavior %behavior_index.

        :param behavior_index:
        :return: Tuple of (observations, rewards, dones). Each element is ndarray[num_agents, *]
        """
        num_agents = self.num_agents_list[behavior_index]
        behavior_spec = self.behavior_specs[behavior_index]
        behavior_name = self.behavior_names[behavior_index]
        # TODO: implement stacked observations:
        observations = np.ndarray((num_agents, *behavior_spec.observation_specs[0].shape))
        rewards = np.ndarray((num_agents, 1))
        dones = np.ndarray((num_agents, 1))
        decision_steps, terminal_steps = self.env.get_steps(behavior_name)
        for agent_id in decision_steps:
            observations[agent_id] = decision_steps[agent_id].obs[0]
            rewards[agent_id] = decision_steps[agent_id].reward
            dones[agent_id] = False
        for agent_id in terminal_steps:
            observations[agent_id] = terminal_steps[agent_id].obs[0]
            rewards[agent_id] = terminal_steps[agent_id].reward
            dones[agent_id] = not terminal_steps[agent_id].interrupted
        return observations, rewards, dones

    def close(self):
        self.env.close()
