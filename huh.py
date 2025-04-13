from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from implementation import SupplyChainEnv

env = make_vec_env(lambda: SupplyChainEnv(), n_envs=1)
model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./ppo_logs/")
model.learn(total_timesteps=5000, tb_log_name="test_run")