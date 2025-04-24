from utils import *
from example import example_use_of_gym_env
from gymnasium.envs.registration import register
from minigrid.envs.doorkey import DoorKeyEnv
from known_env import *
from unknown_env import *
from pathlib import Path

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

def doorkey_problem(env_path, known=True, t=300):
    """
    Run DP for either known map problem or unknown map problem.

    Args:
        env: string, env path or env folder
        known: True if run known map problem
        t: number of horizons for DP for unknown map problem

    Returns:
        optimal action sequence
    """
    assert type(env_path) == str, "Please Provide Env Path or Env Folder"
    seq = []
    # Part A: Known Map Problem
    if known:
        # Run DP for specified map
        env, info = load_env(env_path)
        known_env = KnownEnv(env, info)
        seq = known_env.fdp()
    else:
        env, info, env_p = load_random_env(env_path)
        # check policy
        policy_path = Path('output/unknown_sol.npz')
        # run query if policy exist
        if policy_path.exists():
            print("Policy File Exists, Run Query Now")
            unknown_env = UnknownEnv(env,info)
            seq = unknown_env.extract_seq()
        # recompute policy first before query
        else:
            print("Policy File Not Found, Run DP Now (Will take a while)")
            unknown_policy = UnknownPolicy(t)
            unknown_policy.fdp()
            print("Policy Computed, Run Query Now")
            unknown_env = UnknownEnv(env,info)
            seq = unknown_env.extract_seq()
    return seq

def draw_trajct(seq, env):
    """
    Draw Trajectory on env based on action sequence
    Args:
        seq: action sequence
        env: environment
    """
    positions = [env.agent_pos]

    img = env.render()
    for act in seq:
        step(env, act)
        positions.append(env.agent_pos)
    print(positions)

    plt.imshow(img)
    xs, ys = zip(*positions)
    plt.plot(xs, ys, color='blue', marker='o')
    plt.scatter(xs[0], ys[0], color='green', label='Start')
    plt.scatter(xs[-1], ys[-1], color='red', label='End')
    plt.legend()
    plt.axis('off')
    plt.title("Agent Trajectory Overlay")



def partA():
    env_path = "./envs/known_envs/doorkey-6x6-direct.env"
    env, info = load_env(env_path)
    know_env = KnownEnv(env,info)
    seq = know_env.fdp()
    print(seq)
    #draw_gif_from_seq(seq, load_env(env_path)[0])  # draw a GIF & save


def partB():
    #unknown_policy = UnknownPolicy(t=300)
    #unknown_policy.fdp()
    #print('Finish')
    env_folder = "./envs/random_envs"
    env, info, env_p = load_random_env(env_folder)
    unknown_env = UnknownEnv(env, info)
    #seq = unknown_env.extract_optimal_trajectory()
    seq = unknown_env.extract_seq()
    print(seq)
    #draw_trajct(seq, env)
    #env.reset()
    draw_gif_from_seq(seq, env)
    #unknown_env.check_goal((7,3), (2,2), True)


if __name__ == "__main__":
    #example_use_of_gym_env()
    #partA()
    partB()

