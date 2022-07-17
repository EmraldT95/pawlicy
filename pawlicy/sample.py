import pybullet as p
import pybullet_data as pd

import time

client_id = p.connect(p.GUI)
p.setAdditionalSearchPath(pd.getDataPath())

p.setPhysicsEngineParameter(enableConeFriction=0)
plane = p.loadURDF("plane.urdf")
p.setGravity(0, 0, -9.8)
p.setTimeStep(1. / 500)
#p.setDefaultContactERP(0)
#urdfFlags = p.URDF_USE_SELF_COLLISION+p.URDF_USE_SELF_COLLISION_EXCLUDE_ALL_PARENTS
urdfFlags = p.URDF_USE_SELF_COLLISION
# quadruped = p.loadURDF("laikago/laikago_toes.urdf", [0, 0, .5], [0, 0.5, 0.5, 0],
#                        flags=urdfFlags,
#                        useFixedBase=False)
quadruped = p.loadURDF("a1/a1.urdf", [0, 0, .4], [0, 0, 0, 1],
                       flags=urdfFlags,
                       useFixedBase=False)
#enable collision between lower legs

for j in range(p.getNumJoints(quadruped)):
  print(p.getJointInfo(quadruped, j))

#2,5,8 and 11 are the lower legs
lower_legs = [2, 5, 8, 11]
for l0 in lower_legs:
  for l1 in lower_legs:
    if (l1 > l0):
      enableCollision = 1
      print("collision for pair", l0, l1,
            p.getJointInfo(quadruped, l0)[12],
            p.getJointInfo(quadruped, l1)[12], "enabled=", enableCollision)
      p.setCollisionFilterPair(quadruped, quadruped, 2, 5, enableCollision)

jointIds = []
paramIds = []
jointOffsets = []
jointDirections = [-1, 1, 1, 1, 1, 1, -1, 1, 1, 1, 1, 1]
jointAngles = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

for i in range(4):
  jointOffsets.append(0)
  jointOffsets.append(-0.7)
  jointOffsets.append(0.7)

maxForceId = p.addUserDebugParameter("maxForce", 0, 100, 20)

for j in range(p.getNumJoints(quadruped)):
  p.changeDynamics(quadruped, j, linearDamping=0, angularDamping=0)
  info = p.getJointInfo(quadruped, j)
  #print(info)
  jointName = info[1]
  jointType = info[2]
  if (jointType == p.JOINT_PRISMATIC or jointType == p.JOINT_REVOLUTE):
    jointIds.append(j)

p.getCameraImage(480, 320)
p.setRealTimeSimulation(0)

joints = []

# with open(pd.getDataPath() + "/laikago/data1.txt", "r") as filestream:
#   for line in filestream:
#     print("line=", line)
#     maxForce = p.readUserDebugParameter(maxForceId)
#     currentline = line.split(",")
#     #print (currentline)
#     #print("-----")
#     frame = currentline[0]
#     t = currentline[1]
#     #print("frame[",frame,"]")
#     joints = currentline[2:14]
#     #print("joints=",joints)
#     for j in range(12):
#       targetPos = float(joints[j])
#       p.setJointMotorControl2(quadruped,
#                               jointIds[j],
#                               p.POSITION_CONTROL,
#                               jointDirections[j] * targetPos + jointOffsets[j],
#                               force=maxForce)
#     p.stepSimulation()
#     for lower_leg in lower_legs:
#       #print("points for ", quadruped, " link: ", lower_leg)
#       pts = p.getContactPoints(quadruped, -1, lower_leg)
#       #print("num points=",len(pts))
#       #for pt in pts:
#       # print(pt[9])
#     time.sleep(1. / 500.)

index = 0
for j in range(p.getNumJoints(quadruped)):
  p.changeDynamics(quadruped, j, linearDamping=0, angularDamping=0)
  info = p.getJointInfo(quadruped, j)
  js = p.getJointState(quadruped, j)
  #print(info)
  jointName = info[1]
  jointType = info[2]
  if (jointType == p.JOINT_PRISMATIC or jointType == p.JOINT_REVOLUTE):
    paramIds.append(p.addUserDebugParameter(jointName.decode("utf-8"), -4, 4,
                                (js[0] - jointOffsets[index]) / jointDirections[index]))
    index = index+1
                       

p.setRealTimeSimulation(1)

import matplotlib.pyplot as plt
import numpy as np

while (1):

  for i in range(len(paramIds)):
    c = paramIds[i]
    targetPos = p.readUserDebugParameter(c)
    maxForce = p.readUserDebugParameter(maxForceId)
    p.setJointMotorControl2(quadruped,
                            jointIds[i],
                            p.POSITION_CONTROL,
                            jointDirections[i] * targetPos + jointOffsets[i],
                            force=maxForce)

    rendered_img = None
    if rendered_img is None:
      rendered_img = plt.imshow(np.zeros((100, 100, 4)))

      # Base information
      proj_matrix = p.computeProjectionMatrixFOV(fov=60, aspect=1, nearVal=0.01, farVal=100)
      pos, ori = [list(l) for l in p.getBasePositionAndOrientation(quadruped, client_id)]
      ori = [0, 0, 0, 1]
      # print(pos, '--------------------', ori)
      pos[2] = 0.2

      # Rotate camera direction
      rot_mat = np.array(p.getMatrixFromQuaternion(ori)).reshape(3, 3)
      camera_vec = np.matmul(rot_mat, [1, 0, 0])
      up_vec = np.matmul(rot_mat, np.array([0, 0, 1]))
      view_matrix = p.computeViewMatrix(pos, pos + camera_vec, up_vec)

      # Display image
      frame = p.getCameraImage(100, 100, view_matrix, proj_matrix)[2]
      # frame = np.reshape(frame, (100, 100, 4))
      # rendered_img.set_data(frame)
      # plt.draw()
      # plt.pause(.00001)

terrain = env.world_dict["ground"]
    action_low, action_high = env.action_space.low, env.action_space.high
    action_median = (action_low + action_high) / 2.
    dim_action = action_low.shape[0]
    action_selector_ids = []
    for dim in range(dim_action):
        action_selector_id = p.addUserDebugParameter(paramName="{}".format(constants.JOINT_NAMES[dim]),
                                                    rangeMin=action_low[dim],
                                                    rangeMax=action_high[dim],
                                                    startValue=constants.INIT_MOTOR_ANGLES[dim])
        action_selector_ids.append(action_selector_id)
    p.addUserDebugParameter("reset", 0, 1, 0)

    if FLAGS.video_dir:
        log_id = p.startStateLogging(p.STATE_LOGGING_VIDEO_MP4, FLAGS.video_dir)

    # for _ in tqdm(range(500)):
    try:
        # num_joints = env._pybullet_client.getNumJoints(env.robot.quadruped)
        # _joint_name_to_id = {}
        # for i in range(num_joints):
        #     joint_info = env._pybullet_client.getJointInfo(env.robot.quadruped, i)
        #     _joint_name_to_id[joint_info[1].decode("UTF-8")] = joint_info[0]
        # print(_joint_name_to_id)
        while(1):
            # env.render()
            action = np.ones(dim_action)
            for dim in range(dim_action):
                action[dim] = env.pybullet_client.readUserDebugParameter(action_selector_ids[dim])
            # env.step(env.action_space.sample())
            env.step(action)
    # env._robot.getCameraImage()
    except ValueError:
        env.close()    

        # ground = env.get_ground()
        # print(env.pybullet_client.getContactPoints(bodyA=env._robot.quadruped, bodyB=ground["id"]))

    

    if FLAGS.video_dir:
        p.stopStateLogging(log_id)