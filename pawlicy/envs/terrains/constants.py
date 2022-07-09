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
    'mounts': [0, 0, .85],
    'plane': [0, 0, 0.21],
    # 'hills': [0, 0, 1.98],
    'maze': [0, 0, 0.21],
    'random': [0, 0, 0.35]
}

TERRAIN_INIT_POSITION = {
    'mounts': [0, 0, 2],
    'plane': [0, 0, 0],
    # 'hills': [0, 0, 1.98],
    'maze': [0, 0, 0],
    'random': [0, 0, 0]
}

MESH_SCALES = {
    'mounts': [.1, .1, 24],
    'maze': [0.5, 0.5, 0.5],
    'random': [.05, .05, 1]
}