# From rex_gym

TYPES = [
    'mounts',
    'maze',
    'random',
    'plane'
]

FLAG_TO_FILENAME = {
    'mounts': "heightmaps/wm_height_out.png",
    'maze': "heightmaps/Maze.png"
}

ROBOT_INIT_POSITION = {
    'mounts': [0, 0, 0.32],
    'plane': [0, 0, 0.32],
    # 'hills': [0, 0, 1.98],
    'maze': [0, 0, 0.32],
    'random': [0, 0, 0.42]
}

TERRAIN_INIT_POSITION = {
    'mounts': [0, 0, 0.44],
    'plane': [0, 0, 0],
    # 'hills': [0, 0, 1.98],
    'maze': [0, 0, 0.04],
    'random': [0, 0, 0]
}

MESH_SCALES = {
    'mounts': [.1, .1, 8],
    'maze': [.3, .3, .1],
    'random': [.1, .1, .3]
}