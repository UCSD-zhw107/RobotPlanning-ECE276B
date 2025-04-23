from utils import *
from example import example_use_of_gym_env
from gymnasium.envs.registration import register
from minigrid.envs.doorkey import DoorKeyEnv
from known_env import *
from unknown_env import *

MF = 0  # Move Forward
TL = 1  # Turn Left
TR = 2  # Turn Right
PK = 3  # Pickup Key
UD = 4  # Unlock Door

class DoorKey10x10Env(DoorKeyEnv):
    def __init__(self, **kwargs):
        super().__init__(size=10, **kwargs)

register(
    id='MiniGrid-DoorKey-10x10-v0',
    entry_point='__main__:DoorKey10x10Env'
)

def doorkey_problem(env):
    """
    You are required to find the optimal path in
        doorkey-5x5-normal.env
        doorkey-6x6-normal.env
        doorkey-8x8-normal.env

        doorkey-6x6-direct.env
        doorkey-8x8-direct.env

        doorkey-6x6-shortcut.env
        doorkey-8x8-shortcut.env

    Feel Free to modify this fuction
    """
    optim_act_seq = [TL, MF, PK, TL, UD, MF, MF, MF, MF, TR, MF]
    return optim_act_seq


def partA():
    env_path = "./envs/known_envs/doorkey-6x6-direct.env"
    know_env = KnownEnv(env_path)
    seq = know_env.fdp()
    print(seq)
    #draw_gif_from_seq(seq, load_env(env_path)[0])  # draw a GIF & save


def partB():
    #env_folder = "./envs/random_envs"
    #env, info, env_path = load_random_env(env_folder)
    unknown_env = UnknownEnv()
    print(len(unknown_env.state))
    #unknown_env.fdp()
    print('Finish')


if __name__ == "__main__":
    #example_use_of_gym_env()
    #partA()
    partB()

