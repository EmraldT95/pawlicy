from . import constants
import numpy as np
from .terrain import Terrain

class TerrainRandomizer:
    '''Randomly generates a terrain for the gym envrionment'''
    def __init__(self, pybullet_client, columns=256, rows=256):
        self._pybullet_client = pybullet_client
        self._types = np.array(constants.TYPES)
        self._columns = columns
        self._rows = rows

    def randomize(self):
        terrainType = np.random.choice(self._types, 1)[0]
        terrain = Terrain(pybullet_client=self._pybullet_client,
            type=terrainType,
            columns=self._columns,
            rows=self._rows)
        terrain_id = terrain.generate_terrain()
        return terrain_id