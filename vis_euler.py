
  
import numpy as np
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def plot_axes(ax, origin, rotation, label, color, axis_length=0.3):
    # Define the original coordinate axes
    axes = np.eye(3)

    # Rotate the axes
    rotated_axes = rotation.apply(axes)

    # Plot each axis with different color
    for i, axis in enumerate(rotated_axes):
        ax.quiver(*origin, *axis, color=color[i], length=axis_length)
        ax.text(*(origin + axis), f'{label}{i}', color=color[i], fontsize=12)

def transform_extrinsics(roll, pitch, yaw):
    # Define the extrinsic parameters in radians
    roll = np.deg2rad(roll)
    pitch = np.deg2rad(pitch)
    yaw = np.deg2rad(yaw)

    # Create a rotation object from roll, pitch, yaw (ISO 8855 to your camera's coordinate system)
    r_iso = R.from_euler('xyz', [roll, pitch, yaw])

    # Define the transformation from ISO 8855 to NuScenes:
    # Rotate yaw by -90 degrees and roll by -90 degrees
    R_z = R.from_euler('z', -90, degrees=True)
    R_x = R.from_euler('x', -90, degrees=True)

    # Combine the transformations
    combined_rotation = R_z * R_x

    # Apply the combined rotation to the original extrinsics
    transformed_rotation = combined_rotation * r_iso

    return r_iso, transformed_rotation


# #Example : front:wide:120 extrinsic parameters in degrees
# roll = 0.7478020191192627
# pitch =  0.39888936281204224
# yaw = 1.69412100315094

#Example : front:left:120 extrinsic parameters in degrees
roll = 0.6823626160621643
pitch =  0.6535221934318542
yaw = 67.79759216308594
#Get the original and transformed rotations
r_iso, transformed_rotation = transform_extrinsics(roll, pitch, yaw)
# Visualization
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
# Plot the original coordinate system (ISO 8855)
plot_axes(ax, origin=[0, 0, 0], rotation=r_iso, label='ISO', color=['r', 'g', 'b'], axis_length=0.04)
# Plot the transformed coordinate system (NuScenes) c:qingse y:huangse k:heise
plot_axes(ax, origin=[0, 0, 0], rotation=transformed_rotation, label='NuS', color=['c', 'y', 'k'], axis_length=0.03)
# Set labels
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
#Set the aspect ratio to be equal
ax.set_box_aspect([1,1,1])
plt.show()