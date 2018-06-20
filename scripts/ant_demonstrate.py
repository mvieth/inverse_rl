import tensorflow as tf

from inverse_rl.algos.trpo import TRPO
from inverse_rl.models.tf_util import get_session_config
from sandbox.rocky.tf.policies.gaussian_mlp_policy import GaussianMLPPolicy
from sandbox.rocky.tf.envs.base import TfEnv
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline

from inverse_rl.envs.env_utils import CustomGymEnv
from inverse_rl.utils.log_utils import rllab_logdir
from inverse_rl.utils.hyper_sweep import run_sweep_parallel, run_sweep_serial

#Loads a policy from the given pickle-file and records a video
if __name__ == "__main__":
    #filename='data/ant_data_collect/2018_05_25_13_42_59_0/itr_1499.pkl'
    #filename='data/ant_data_collect/2018_05_23_15_21_40_0/itr_1499.pkl'
    #filename='data/ant_data_collect/2018_05_19_07_56_37_1/itr_1499.pkl'
    #filename='data/ant_data_collect/2018_05_19_07_56_37_0/itr_1485.pkl'
    #filename='data/ant_state_irl/2018_05_26_08_51_16_0/itr_999.pkl'
    #filename='data/ant_state_irl/2018_05_26_08_51_16_1/itr_999.pkl'
    #filename='data/ant_state_irl/2018_05_26_08_51_16_2/itr_999.pkl'
    filename='data/ant_transfer/2018_05_26_16_06_05_4/itr_999.pkl'
    import gym
    import joblib
    import rllab.misc.logger as rllablogger
    tf.reset_default_graph()
    with tf.Session(config=get_session_config()) as sess:
        rllablogger.set_snapshot_dir("data/video")
        saved=joblib.load(filename)
        env = TfEnv(CustomGymEnv('CustomAnt-v0', record_video=True, record_log=True))#'DisabledAnt-v0' #Switch for the DisabledAnt for the transfer task
        policy=saved['policy']
        observation = env.reset()
        for _ in range(1000):
            env.render()
            action,rest = policy.get_action(observation)
            observation, reward, done, info = env.step(action)
