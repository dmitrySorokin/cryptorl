from gym.envs.registration import register
from .envs import CryptoEnv

register(id='cryptorl-v0',
         entry_point='cryptorl.envs:CryptoEnv')
