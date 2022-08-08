from locomotion.robots.a1 import A1

import numpy as np

URDF_FILENAME = "a1/a1.urdf"

class A1_loco(A1):
    def __init__(self, 
        pybullet_client, 
        urdf_filename=URDF_FILENAME,
        enable_clip_motor_commands=False, 
        time_step=0.001, 
        action_repeat=10, 
        sensors=None, 
        control_latency=0.002, 
        on_rack=False, 
        enable_action_interpolation=True, 
        enable_action_filter=False, 
        motor_control_mode=None, 
        reset_time=1, 
        allow_knee_contact=False):


        super().__init__(pybullet_client, urdf_filename, enable_clip_motor_commands, 
                        time_step, action_repeat, sensors, control_latency, on_rack, 
                        enable_action_interpolation, enable_action_filter, motor_control_mode, 
                        reset_time, allow_knee_contact)
    
    def GetRawBaseOrientation(self):
        """
        Returns raw absolute orientation of base in quaternions
        """
        orientation = self._pybullet_client.getBasePositionAndOrientation(self.quadruped)[1]
        return orientation

    def GetRawBaseRollPitchYaw(self):
        """
        Returns raw absolute orientation of base in euler
        """
        roll_pitch_yaw = self._pybullet_client.getEulerFromQuaternion(self.GetRawBaseOrientation())
        return np.asarray(roll_pitch_yaw)
    