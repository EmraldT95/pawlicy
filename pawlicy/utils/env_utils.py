from tabulate import tabulate
import logging
import os
import copy
import math

logger = logging.getLogger()

def _set_log_lvl(log_level):
    logging.basicConfig(filename="a1_env.log", filemode='w', level=os.environ.get("LOGLEVEL", log_level))

def _log(desc="", msg="", level="DEBUG", is_tabular=True, header="firstrow"):

    headers = {
        "joint_infos": ["jointIndex","jointName","jointType", "qIndex", "uIndex", "flags", 
                        "jointDamping", "jointFriction", "jointLowerLimit", "jointUpperLimit", 
                        "jointMaxForce", "jointMaxVelocity", "linkName", "jointAxis", "parentFramePos", 
                        "parentFrameOrn", "parentIndex"
                        ],
        "joint_states": ["jointPosition", "jointVelocity", "jointReactionForces", 
                        "appliedJointMotorTorque"
                        ],
        "shape_infos" : ["objectUniqueId", "linkIndex", "visualGeometryType", "dimensions", 
                        "meshAssetFileName", "localVisualFrame position", "localVisualFrame orientation",
                        "rgbaColor", "textureUniqueId"
                        ],
        "link_states" : ["linkWorldPosition", "linkWorldOrientation", "localInertialFramePosition", 
                        "localInertialFrameOrientation", "worldLinkFramePosition", "worldLinkFrameOrientation",
                        "worldLinkLinearVelocity", "worldLinkAngularVelocity",
                        ],
        "link_angular_velocities" : ["dR", "dP", "dY"]
    }

    if is_tabular:
        logger.debug(f"{desc} : \n{tabulate(msg, headers=headers[header])}")
    else:
        logger.debug(f"{desc} : {msg}")


def MapToMinusPiToPi(angle):
    """Maps as angle to [-pi, pi].

    Args:
      angle: The angle in rad.
    """
    mapped_angle = math.fmod(angle, 2 * math.pi)
    if mapped_angle >= math.pi:
        mapped_angle -= 2 * math.pi
    elif mapped_angle < -math.pi:
        mapped_angle += 2 * math.pi
    return mapped_angle
