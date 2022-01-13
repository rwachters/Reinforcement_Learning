from my_unity_environment import MyUnityEnvironment
import numpy as np
import concurrent
from concurrent.futures import ThreadPoolExecutor, Future
from typing import Tuple, List, Optional, Any


class ParallelUnityEnvironment:
    def __init__(self, num_envs: int, seeds: List[int], file_name=None, no_graphics=False, worked_id_start=0):
        """
        :param num_envs: number of environments to run in parallel
        :param seeds: a list of random seeds for each environment
        :param file_name: The filename of the Unity executable, or None when using the Unity editor
            (press Play to connect).
        :param no_graphics: Whether to use graphics windows or not.
        :param worked_id_start: The id of the first Unity thread to create.
          For example, a value of 4 would create threads with ids: 4, 5, 6 etc.
        """
        if len(seeds) != num_envs:
            raise ValueError()

        def _init_env(_file_name, _no_graphics, _seed, _worker_id):
            return MyUnityEnvironment(file_name=_file_name, no_graphics=_no_graphics, seed=_seed, worker_id=_worker_id)

        self.num_envs = num_envs
        self.executor = ThreadPoolExecutor(max_workers=num_envs + 2, thread_name_prefix="Unity_")
        self.futures: List[Future[Any]] = [
            self.executor.submit(_init_env, file_name, no_graphics, seed, worker_id)
            for seed, worker_id in zip(seeds, range(worked_id_start, worked_id_start + num_envs))]
        self.envs: List[MyUnityEnvironment] = [future.result() for future in self.futures]
        self.behavior_names = self.envs[0].behavior_names
        self.behavior_specs = self.envs[0].behavior_specs
        self.num_agents_list = self.envs[0].num_agents_list

    def set_timescale(self, time_scale: float):
        """ Set the timescale at which the physics simulation runs.

        :param time_scale: a value of 1.0 means the simulation runs in realtime.
        """
        for env in self.envs:
            env.set_timescale(time_scale=time_scale)

    def set_display_size(self, width: int, height: int):
        for env in self.envs:
            env.set_display_size(width=width, height=height)

    def reset(self, reset_list: List[bool]):
        """Resets all environments where reset_list[env_index] == True """

        def _reset(env: MyUnityEnvironment):
            env.reset()

        for env_index, reset in enumerate(reset_list):
            if reset:
                self.futures[env_index] = self.executor.submit(_reset, self.envs[env_index])
        concurrent.futures.wait(self.futures)

    def get_observations(self, behavior_index: int, env_index: int):
        """ Get observations for each environment.

        :return: np.ndarray[num_agents, observation_size]"""

        return self.envs[env_index].get_observations(behavior_index)

    def set_actions(self, behavior_index: int, env_index: int, continuous: Optional[np.ndarray] = None,
                    discrete: Optional[np.ndarray] = None):
        self.envs[env_index].set_actions(behavior_index, continuous, discrete)

    def get_experiences(self, behavior_index: int, env_index: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """ Get experiences for environment %env_index with behavior %behavior_index.

        :param behavior_index:
        :param env_index:
        :return: Tuple of (observations, rewards, dones). Each element is ndarray[num_agents, *]
        """
        return self.envs[env_index].get_experiences(behavior_index)

    def step(self):
        """ Step forward in all environments."""

        def _step(myenv: MyUnityEnvironment):
            myenv.step()

        for env_index, env in enumerate(self.envs):
            self.futures[env_index] = self.executor.submit(_step, env)
        concurrent.futures.wait(self.futures)

    def close(self):
        def _close(_env: MyUnityEnvironment):
            _env.close()

        for env_index, env in enumerate(self.envs):
            self.futures[env_index] = self.executor.submit(_close, env)
        concurrent.futures.wait(self.futures)
        self.executor.shutdown()
