# From rex_gym

TYPES = [
    'mounts',
    'maze',
    'random'
]

FLAG_TO_FILENAME = {
    'mounts': "heightmaps/wm_height_out.png",
    'maze': "heightmaps/Maze.png"
}

ROBOT_INIT_POSITION = {
    'mounts': [0, 0, 0],
    'plane': [0, 0, 0.21],
    'maze': [0, 0, 0.21],
    'random': [0, 0, 0]
}

MESH_SCALES = {
    'mounts': [.1, .1, 24],
    'maze': [1, 1, 1],
    'random': [.05, .05, 1]
}