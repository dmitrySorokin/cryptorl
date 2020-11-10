import gym
import numpy as np
from .utils import Environment


class CryptoEnv(gym.Env):
    n_actions = 4
    _max_episode_steps = 100
    _state_len = 8
    _offset = _state_len + n_actions
    _obs_len = _offset * _max_episode_steps

    observation_space = gym.spaces.Box(low=-1, high=1, shape=(_obs_len, ), dtype=np.float32)
    action_space = gym.spaces.Box(low=-1, high=1, shape=(n_actions,), dtype=np.float32)

    first_click = np.array([[1], [0]])
    second_click = np.array([[0], [1]])

    def __init__(self):
        self.impl = Environment()
        self.step_id = None
        self.handles = None
        self.state = None

    def step(self, action):
        self.handles = np.clip(self.handles + action, -1, 1)
        delta_0_0, delta_pi2_pi2, delta_pi_0, delta_3pi2_pi2 = self._calc_state()
        loss = np.mean(list(map(np.linalg.norm, [delta_0_0, delta_pi2_pi2, delta_pi_0, delta_3pi2_pi2])))
        deltas = np.concatenate([delta_0_0, delta_pi2_pi2, delta_pi_0, delta_3pi2_pi2], axis=0).reshape(-1)
        self.state[self.step_id * CryptoEnv._offset: (self.step_id + 1) * CryptoEnv._offset] = [*deltas, *action]
        self.step_id += 1
        info = {
            'loss': loss,
            'delta_0_0': np.linalg.norm(delta_0_0),
            'delta_pi2_pi2': np.linalg.norm(delta_pi2_pi2),
            'delta_pi_0': np.linalg.norm(delta_pi_0),
            'delta_3pi2_pi2': np.linalg.norm(delta_3pi2_pi2),
            'delta_id': self.step_id
        }
        return self.state, 1 - loss, self.step_id == CryptoEnv._max_episode_steps, info

    def reset(self):
        self.impl.InitRandom()
        self.step_id = 0
        self.handles = np.zeros(CryptoEnv.action_space.shape)
        self.state = np.zeros(CryptoEnv.observation_space.shape)
        return self.step(self.handles)[0]

    def _calc_state(self):
        rescaled = self._rescale_action(self.handles)
        return (
            self.impl.Step(0, 0, *rescaled) - CryptoEnv.first_click,
            self.impl.Step(np.pi / 2, np.pi / 2, *rescaled) - CryptoEnv.second_click,
            self.impl.Step(np.pi, 0, *rescaled) - CryptoEnv.second_click,
            self.impl.Step(3 * np.pi / 2, np.pi / 2, *rescaled) - CryptoEnv.first_click
        )

    def _rescale_action(self, action):
        """
        :param action: [-1, 1]
        :return: [0, 2 * pi]
        """
        return (np.asarray(action) + 1) * np.pi
