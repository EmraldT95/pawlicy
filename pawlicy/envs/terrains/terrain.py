from . import constants

import pybullet as p
import pybullet_data as pd
import numpy as np
from perlin_noise import PerlinNoise
import random

BASE_ORIENTATION = [0, 0, 0, 1]

class Terrain:

    def __init__(self, pybullet_client, type="random", columns=256, rows=256):
        self._pybullet_client = pybullet_client
        self._type = type
        self._columns = columns
        self._rows = rows
    
    def generate_terrain(self):
        self._pybullet_client.setAdditionalSearchPath(pd.getDataPath())
        self._pybullet_client.configureDebugVisualizer(self._pybullet_client.COV_ENABLE_RENDERING, 0)

        # Check whether we want one the pre-existing terrain data in pybullet or to generate a random one
        if self._type == "random":
            terrain_data = [0] * self._columns * self._rows
            noise = PerlinNoise(octaves=25, seed=random.random())
            for j in range(int(self._columns / 2)):
                for i in range(int(self._rows / 2)):
                    height = noise([j/self._columns,i/self._rows]) # Creates Perlin noise for smooth terrain generation
                    # height = random.uniform(0, height_perturbation_range)
                    terrain_data[2 * i + 2 * j * self._rows] = height
                    terrain_data[2 * i + 1 + 2 * j * self._rows] = height
                    terrain_data[2 * i + (2 * j + 1) * self._rows] = height
                    terrain_data[2 * i + 1 + (2 * j + 1) * self._rows] = height

            # Create the terrain in the simulator
            flags = p.GEOM_CONCAVE_INTERNAL_EDGE
            terrain_shape = self._pybullet_client.createCollisionShape(
                shapeType=self._pybullet_client.GEOM_HEIGHTFIELD,
                meshScale=constants.MESH_SCALES[self._type],
                flags=flags,
                heightfieldTextureScaling=(self._rows - 1) / 2,
                heightfieldData=terrain_data,
                numHeightfieldRows=self._rows,
                numHeightfieldColumns=self._columns)
            terrain_id = self._pybullet_client.createMultiBody(0, terrain_shape)

        # For plane, just load the URDF file
        elif self._type == "plane":
            terrain_id = self._pybullet_client.loadURDF("plane_implicit.urdf")

        # This is mounts and maze types
        else:
            file_location = constants.FLAG_TO_FILENAME[self._type]
            if not file_location:
                raise ValueError("Terrain of type %s was not found." % self._type)
            else:
                terrain_shape = self._pybullet_client.createCollisionShape(
                    shapeType=self._pybullet_client.GEOM_HEIGHTFIELD,
                    meshScale=constants.MESH_SCALES[self._type],
                    fileName=file_location)
                terrain_id = self._pybullet_client.createMultiBody(0, terrain_shape) # Mass, Unique ID from createCollisionShape
                # For the "mounts" terrain type, pybullet has a texture file
                if self._type == "mounts":
                    textureId = self._pybullet_client.loadTexture("heightmaps/gimp_overlay_out.png")
                    self._pybullet_client.changeVisualShape(terrain_id, -1, textureUniqueId=textureId, rgbaColor=[1, 1, 1, 1])
                else:
                    self._pybullet_client.changeVisualShape(terrain_id, -1, rgbaColor=[0, 1, 1, 1])

        # Set the position and orientation of the terrain in the simulator
        self._pybullet_client.resetBasePositionAndOrientation(terrain_id, constants.TERRAIN_INIT_POSITION[self._type], BASE_ORIENTATION)
        self._pybullet_client.configureDebugVisualizer(self._pybullet_client.COV_ENABLE_RENDERING, 1)
        return terrain_id, self._type
