import numpy as np

# Possible Directions
RIGHT = (1,0)
LEFT = (-1,0)
UP = (0,-1)
DOWN = (0,1)
DIR_MAP = {
    (1, 0): 0,   # RIGHT
    (0, 1): 1,   # DOWN
    (0, -1): 2,  # UP
    (-1, 0): 3   # LEFT
}
DIR_LIST = [(1, 0), (0, 1), (0, -1), (-1, 0)]

# TL map
TL_MAP = {
    RIGHT:UP,
    UP:LEFT,
    LEFT:DOWN,
    DOWN:RIGHT
}
# TR map
TR_MAP = {
    RIGHT:DOWN,
    DOWN:LEFT,
    LEFT:UP,
    UP:RIGHT
}

# DOOR position
DOOR_POS = ((5,3), (5,7))

# WALL position
WALL_POS = ((5,0), (5,1), (5,2), (5,4), (5,5), (5,6), (5,8), (5,9))

# KEY position
KEY_POSITION = [(2,2), (2,3), (1,6)]

# GOAL position
GOAL_POSITION = [(6, 1), (7, 3), (6, 6)]

# Initial agent Pose
INIT_POS = (4,8)
INIT_DIR = UP

# T = number of states
NUM_STATE = 92 * 4 * 2 * 2 * 2 * 3 * 3

# Height and Width
HEIGHT = 10
WIDTH = 10


def encode_node(node):
    t, pos, dir, key, door1, door2, key_pos, goal_pos = node
    x, y = pos
    dx, dy = dir
    dir_idx = DIR_MAP[(dx, dy)]
    key_idx = int(key)
    door1_idx = int(door1)
    door2_idx = int(door2)
    key_pos_idx = KEY_POSITION.index(key_pos)
    goal_pos_idx = GOAL_POSITION .index(goal_pos)

    index = (
        t * (100 * 4 * 2 * 2 * 2 * 3 * 3) +
        (x * 10 + y) * (4 * 2 * 2 * 2 * 3 * 3) +
        dir_idx * (2 * 2 * 2 * 3 * 3) +
        key_idx * (2 * 2 * 3 * 3) +
        door1_idx * (2 * 3 * 3) +
        door2_idx * (3 * 3) +
        key_pos_idx * 3 +
        goal_pos_idx
    )
    return index


def decode_node(index):
    base = index

    goal_pos_idx = base % 3
    base //= 3
    key_pos_idx = base % 3
    base //= 3
    door2_idx = base % 2
    base //= 2
    door1_idx = base % 2
    base //= 2
    key_idx = base % 2
    base //= 2
    dir_idx = base % 4
    base //= 4
    pos_idx = base % 100
    base //= 100
    t = base

    x = pos_idx // 10
    y = pos_idx % 10
    dx, dy = DIR_LIST[dir_idx]
    key_pos = KEY_POSITION[key_pos_idx]
    goal_pos = GOAL_POSITION[goal_pos_idx]
    node = (
        t,
        (x, y),
        (dx, dy),
        key_idx,
        door1_idx,
        door2_idx,
        key_pos,
        goal_pos
    )
    return node

ORI = np.array([
            [1,0],
            [0,1],
            [0,-1],
            [-1,0]
        ])
node = (3, (4, 7), (0, -1), 1, 0, 1, (2, 2), (6, 1))
idx = encode_node(node)
recovered = decode_node(idx)
assert node == recovered
