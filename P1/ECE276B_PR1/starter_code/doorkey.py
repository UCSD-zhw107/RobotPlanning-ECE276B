from utils import *
from example import example_use_of_gym_env
from gymnasium.envs.registration import register
from minigrid.envs.doorkey import DoorKeyEnv
from known_env import *
from unknown_env import *
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from minigrid.core.world_object import Wall, Goal, Key, Door

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

def doorkey_problem(env, info, env_path, known=True, t=300):
    """
    Run DP for either known map problem or unknown map problem.

    Args:
        env: string, env path or env folder
        known: True if run known map problem
        t: number of horizons for DP for unknown map problem

    Returns:
        optimal action sequence
    """
    assert type(env_path) == str, "Please Provide Env Path"
    seq = []
    # Part A: Known Map Problem
    if known:
        # Run DP for specified map
        known_env = KnownEnv(env, info)
        seq = known_env.fdp()
        print(f'Action Sequence: {seq}')
        draw_traj(seq, env, get_env_name(env_path))
    # Part B: Known Map Problem
    else:
        # check policy
        policy_path = Path('./output/unknown_sol.npz')
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
        print(f'Action Sequence: {seq}')
        draw_traj(seq, env, get_env_name(env_path))
    return seq


def draw_traj(seq, env, env_name):
    """
    Given a MiniGrid env, draw all static background: walls, goal, key, doors.
    """
    gif_path = f'./gif/{env_name}.gif'
    traj_path = f'./traj/{env_name}_traj.png'

    grid = env.grid
    width = grid.width
    height = grid.height

    wall_pos_list = []
    key_pos_list = []
    door_pos_list = []
    goal_pos_list = []

    positions = [env.agent_pos]
    with imageio.get_writer(gif_path, mode="I", duration=0.8) as writer:
        img = env.render()
        writer.append_data(img)
        for act in seq:
            step(env, act)
            positions.append(env.agent_pos)
            img = env.render()
            writer.append_data(img)
    print(f"GIF is written to {gif_path}")
    for x in range(width):
        for y in range(height):
            obj = grid.get(x, y)
            if isinstance(obj, Wall):
                wall_pos_list.append((x, y))
            elif isinstance(obj, Key):
                key_pos_list.append((x, y))
            elif isinstance(obj, Door):
                door_pos_list.append((x, y))
            elif isinstance(obj, Goal):
                goal_pos_list.append((x, y))
    fig, ax = plt.subplots(figsize=(width, height))
    ax.set_xlim(-0.5, width - 0.5)
    ax.set_ylim(-0.5, height - 0.5)
    ax.set_aspect('equal')
    ax.grid(True)
    ax.invert_yaxis()
    for x, y in wall_pos_list:
        rect = patches.Rectangle((x - 0.5, y - 0.5), 1, 1, color='black')
        ax.add_patch(rect)
    for x, y in key_pos_list:
        ax.scatter(x, y, marker='*', color='red', s=500, label='Key')
    for x, y in door_pos_list:
        ax.scatter(x, y, marker='s', color='brown', s=300, label='Door')

    for x, y in goal_pos_list:
        ax.scatter(x, y, marker='P', color='green', s=500, label='Goal')

    xs, ys = zip(*positions)
    ax.plot(xs, ys, color='blue', marker='o', linewidth=2, markersize=6, label='Agent Trajectory')

    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys())

    plt.title("Trajectory of Door Key")
    plt.savefig(traj_path)
    plt.show()
    print(f"Trajectory is written to {traj_path}")

def get_env_name(path):
    return Path(path).stem


def partA(env_path):
    """
    Known Map Problem
    Args:
        env_path: specified env path
    """
    env, info = load_env(env_path)
    doorkey_problem(env, info, env_path, known=True)


def partB(env_folder):
    """
    Unknown Map Problem, will load random env
    Args:
        env_folder: env folder.
    """
    env, info, env_path = load_random_env(env_folder)
    doorkey_problem(env, info, env_path, known=False)



if __name__ == "__main__":
    # Please Provide a path to .env file for any known map
    #partA('./envs/known_envs/doorkey-8x8-shortcut.env')

    # Please Provide foler path to random env, it will only load random map
    partB(env_folder='./envs/random_envs')

