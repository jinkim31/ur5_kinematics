import ur5_kinematics
import numpy as np
import matplotlib.pyplot as plt


def plot_transforms(tfs):
    ax = plt.figure().add_subplot(projection='3d')
    ax.quiver(tfs[:, 0, 3], tfs[:, 1, 3], tfs[:, 2, 3], tfs[:, 0, 0], tfs[:, 1, 0], tfs[:, 2, 0], length=0.1, color='r')
    ax.quiver(tfs[:, 0, 3], tfs[:, 1, 3], tfs[:, 2, 3], tfs[:, 0, 1], tfs[:, 1, 1], tfs[:, 2, 1], length=0.1, color='g')
    ax.quiver(tfs[:, 0, 3], tfs[:, 1, 3], tfs[:, 2, 3], tfs[:, 0, 2], tfs[:, 1, 2], tfs[:, 2, 2], length=0.1, color='b')
    ax.set_aspect('equal')


joints = [0, 0, 0, 0, 0, 0]
tfs = np.array([
    np.eye(4),
    ur5_kinematics.tf0to1(joints),
    ur5_kinematics.tf0to2(joints),
    ur5_kinematics.tf0to3(joints),
    ur5_kinematics.tf0to4(joints),
    ur5_kinematics.tf0to5(joints),
    ur5_kinematics.tf0to6(joints),
])

x_d = np.array([0.418, 0.273, 0.264])
joints = np.array([0, 0, 0.1, 0, 0.1, 0]).T

for i in range(100):
    # forward kinematics for current joint state
    x = ur5_kinematics.tf0to6(joints.T)[:3, 3]
    print(x)

    # derive jacobian
    j = ur5_kinematics.position_jacobian(joints.T)

    # find jacobian pseudo inverse
    j_inv = np.dot(j.T, np.linalg.inv(np.dot(j, j.T)))

    # evaluate error
    pos_error = x_d.T - x
    if np.linalg.norm(pos_error) < 1e-4:
        break

    # update joint state
    joints += np.dot(j_inv, pos_error)
