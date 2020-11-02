import gym
import numpy as np
from .utils import Environment


class CryptoEnv(gym.Env):
    n_actions = 4
    episode_len = 100
    # obs = [[action_i1, action_i2, action_i3, action_i4, error_i], ...] where i is step

    observation_space = gym.spaces.Box(low=0, high=2 * np.pi, shape=(episode_len, n_actions + 1), dtype=np.float64)
    action_space = gym.spaces.Box(low=0, high=2 * np.pi, shape=(n_actions,), dtype=np.float64)

    first_click = np.array([[1], [0]])
    second_click = np.array([[0], [1]])

    def __init__(self):
        self.impl = Environment()
        self.step_id = None
        self.state = None

    def step(self, action):
        loss = self._calc_loss(action)
        self.state[self.step_id] = [*action, loss]
        self.step_id += 1
        return self.state, -loss, {}, self.step_id == CryptoEnv.episode_len

    def reset(self):
        self.impl.InitRandom()
        self.step_id = 0
        self.state = np.zeros(CryptoEnv.observation_space.shape)

    def _calc_loss(self, action):
        return np.mean(
            np.linalg.norm(self.impl.Step(0, 0, *action) - CryptoEnv.first_click) +
            np.linalg.norm(self.impl.Step(np.pi / 2, np.pi / 2, *action) - CryptoEnv.second_click) +
            np.linalg.norm(self.impl.Step(np.pi, 0, *action) - CryptoEnv.second_click) +
            np.linalg.norm(self.impl.Step(3 * np.pi / 2, np.pi / 2, *action) - CryptoEnv.first_click)
        )
