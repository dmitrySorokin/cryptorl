import gym
import numpy as np
from .utils import Environment


class CryptoEnv(gym.Env):
    n_actions = 4
    episode_len = 100
    # obs = [[action_i1, action_i2, action_i3, action_i4, error_i], ...] where i is step

    observation_space = gym.spaces.Box(low=-1, high=1, shape=(episode_len, n_actions + 1), dtype=np.float64)
    action_space = gym.spaces.Box(low=-1, high=1, shape=(n_actions,), dtype=np.float64)

    first_click = np.array([[1], [0]])
    second_click = np.array([[0], [1]])

    def __init__(self):
        self.impl = Environment()
        self.step_id = None
        self.state = None
        self.handles = None

    def step(self, action):
        self.handles = np.clip(self.handles + action, -1, 1)
        loss_0_0, loss_pi2_pi2, loss_pi_0, loss_3pi2_pi2 = self._calc_loss(action)
        loss = np.mean([loss_0_0, loss_pi2_pi2, loss_pi_0, loss_3pi2_pi2])
        self.state[self.step_id] = [*action, loss]
        self.step_id += 1
        info = {
            'loss': loss,
            'loss_0_0': loss_0_0,
            'loss_pi2_pi2': loss_pi2_pi2,
            'loss_pi_0': loss_pi_0,
            'loss_3pi2_pi2': loss_3pi2_pi2,
            'step_id': self.step_id
        }
        return self.state, -loss, self.step_id == CryptoEnv.episode_len, info

    def reset(self):
        self.impl.InitRandom()
        self.step_id = 0
        self.handles = np.zeros(CryptoEnv.action_space.shape)
        self.state = np.zeros(CryptoEnv.observation_space.shape)
        return self.step(self.handles)[0]

    def _calc_loss(self, action):
        rescaled = self._rescale_action(action)
        return (
            np.linalg.norm(self.impl.Step(0, 0, *rescaled) - CryptoEnv.first_click),
            np.linalg.norm(self.impl.Step(np.pi / 2, np.pi / 2, *rescaled) - CryptoEnv.second_click),
            np.linalg.norm(self.impl.Step(np.pi, 0, *rescaled) - CryptoEnv.second_click),
            np.linalg.norm(self.impl.Step(3 * np.pi / 2, np.pi / 2, *rescaled) - CryptoEnv.first_click)
        )

    def _rescale_action(self, action):
        """
        :param action: [-1, 1]
        :return: [0, 2 * pi]
        """
        return (np.asarray(action) + 1) * np.pi
