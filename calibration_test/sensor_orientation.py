import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.collections import LineCollection
from mpl_toolkits.mplot3d.art3d import Poly3DCollection  # Imported Poly3DCollection
from draw_sector import *
import argparse

# Jun-2024 Extrinsic parameters of the 6 cameras

def main(args):
    version = args.version
    # Feb params
    if "12-2023" in version:
        #Feb-2024 Extrinsic parameters of the 6 cameras
        title = "Camera layout, with ext-param exported from 12-2023"
        Rots = [
        [-89.32062446116468, 0.07854765653610787, -22.521672904491428],
        [-89.12766650319101, 0.42209469154476525, -88.13024617731571],
        [-88.77181959152223, -0.07233417034149466, -156.85559141635895],
        [-89.56323873996735, -0.10847647488116517, 61.77570351958275],
        [-92.54261630773543, -1.3810911774635233, 89.03523875772953],
        [-88.29269583523273, 1.524097919464106, 118.26059056818484]
        ]
        trans = [
        [2.638101816177368,0.9850038886070251,0.966140866279602],
        [2.132892608642578,-0.012395327910780907,1.5895559787750244],
        [2.642765760421753,-0.9212404489517212,0.968400776386261],
        [2.174730062484741,1.0828661918640137,1.1494275331497192],
        [-0.8957435488700867,0.2945128083229065,1.399481177330017],
        [-0.9570000171661377,-0.8299999833106995,0.8299999833106995]
        ]
        fov = [120, 120, 120, 70, 30, 70]
    elif "6-2023" in version: # Jun trace
        #Jun-2024 Extrinsic parameters of the 6 cameras
        title = "Camera layout, with ext-param exported from 6-2023, better one."

        Rots = [
        [-89.41660334169865, 0.5758636444807045, -22.635946094989784],
        [-89.10048768669368, 0.15787561051547427, -88.08002339303495],
        [-88.93696810305119, -0.34834206476807944, -156.8713299259543],
        [-89.27569949626923, 0.49783222377300884, 61.4024996366352],
        [-92.43325978517534, -0.42888303101061875, 89.05152702331544],
        [-88.59463718719782, 0.6139325350522989, 118.43066567182545]
        ]
        trans = [
        [2.6320180892944336, 0.9325082302093506, 0.9819566011428833],
        [2.1456167697906494, -0.04628324881196022, 1.6052480936050415],
        [2.6182913780212402, -0.9509901404380798, 0.9832093715667725],
        [2.1692914962768555, 1.0484946966171265, 1.1717629432678223],
        [-0.8830921053886414, 0.3092159330844879, 1.3925449848175049],
        [2.1475114822387695, -1.0130553245544434, 1.1678566932678223]
        ]
        fov = [120, 120, 120, 70, 30, 70]
    elif "feb24rots_trans-nusc" in version: # Jun trace
        #Jun-2024 Extrinsic parameters of the 6 cameras
        title = "Camera layout, feb2024's rots params and nuscene's trans params, best one sofar."

        Rots = [
        [-89.41660334169865, 0.5758636444807045, -22.635946094989784],
        [-89.10048768669368, 0.15787561051547427, -88.08002339303495],
        [-88.93696810305119, -0.34834206476807944, -156.8713299259543],
        [-89.27569949626923, 0.49783222377300884, 61.4024996366352],
        [-92.43325978517534, -0.42888303101061875, 89.05152702331544],
        [-88.59463718719782, 0.6139325350522989, 118.43066567182545]
        ]
        trans = [
        [1.52387798135,0.494631336551,1.50932822144],
        [1.70079118954,0.0159456324149,1.51095763913],
        [1.5508477543,-0.493404796419,1.49574800619],
        [1.03569100218,0.484795032713,1.59097014818],
        [0.0283260309358,0.00345136761476,1.57910346144],
        [1.0148780988,-0.480568219723,1.56239545128]
        ]
        
        fov = [120, 30, 120, 70, 30, 70]
    elif "nusc" in version: # Jun trace
        #Jun-2024 Extrinsic parameters of the 6 cameras
        title = "Nuscene orignial calibration for reference"

        Rots = [
        [-89.85977500320001, 0.12143609391200118, -34.839035595600016],
        [-90.32322642770004, -0.04612719483860205, -89.6742843141],
        [-90.78202358850001, 0.5188438566960037, -146.404397903],
        [-90.91736319750001, -0.21518275753700122, 18.600246142799996],
        [-89.0405962694, 0.22919685786400154, 89.86124500000001],
        [-90.93206677999999, 0.6190947610589997, 159.200715506]
        ]
        trans = [
        [1.52387798135,0.494631336551,1.50932822144],
        [1.70079118954,0.0159456324149,1.51095763913],
        [1.5508477543,-0.493404796419,1.49574800619],
        [1.03569100218,0.484795032713,1.59097014818],
        [0.0283260309358,0.00345136761476,1.57910346144],
        [1.0148780988,-0.480568219723,1.56239545128]
        ]
        fov = [70, 70, 70, 70, 120, 70]
    else:
        print(f"unknown version: {version} is given")
        return
    # fov = [120, 120, 120, 70, 30, 70]
    # Convert extrinsic parameters to rotation matrices
    rotations = []
    for roll, pitch, yaw in Rots:
        yaw = np.deg2rad(yaw)
        pitch = np.deg2rad(pitch)
        roll = np.deg2rad(roll)
        rotation = np.array([
        [np.cos(yaw) * np.cos(pitch), np.cos(yaw) * np.sin(pitch) * np.sin(roll) - np.sin(yaw) * np.cos(roll), np.cos(yaw) * np.sin(pitch) * np.cos(roll) + np.sin(yaw) * np.sin(roll)],
        [np.sin(yaw) * np.cos(pitch), np.sin(yaw) * np.sin(pitch) * np.sin(roll) + np.cos(yaw) * np.cos(roll), np.sin(yaw) * np.sin(pitch) * np.cos(roll) - np.cos(yaw) * np.sin(roll)],
        [-np.sin(pitch), np.cos(pitch) * np.sin(roll), np.cos(pitch) * np.cos(roll)]
        ])
        rotations.append(rotation)
    # Set up the 3D plot
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim([-5, 5])
    ax.set_ylim([-5, 5])
    ax.set_zlim([-5, 5])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(f"{title}")
    # Plot the camera orientations
    for i, (rotation, translation, fov_angle, rot) in enumerate(zip(rotations, trans, fov, Rots)):    
        # Define the camera's coordinate axes, rotation is 3*3 rot matrixxxxxx
        x_axis = rotation[:, 0] * 1.5 + translation
        y_axis = rotation[:, 1] * 1.5 + translation
        z_axis = rotation[:, 2] * 1.5 + translation

        # Plot the camera's coordinate axes
        ax.plot([translation[0], x_axis[0]], [translation[1], x_axis[1]], [translation[2], x_axis[2]], color='r', linewidth=2)
        ax.plot([translation[0], y_axis[0]], [translation[1], y_axis[1]], [translation[2], y_axis[2]], color='g', linewidth=2)
        ax.plot([translation[0], z_axis[0]], [translation[1], z_axis[1]], [translation[2], z_axis[2]], color='b', linewidth=2)
    
    # Plot the camera's field of view
        rot_rad = np.deg2rad(rot[-1])
        fov_rad = np.deg2rad(fov_angle)
        # -fov_rad/2 + np.pi/2 + rot_rad ;  + np.pi/2 + rot_rad
        x, y, z = draw_sector(radius=3, start_angle=(0 -fov_rad/2 + rot_rad + np.pi/2 ), stop_angle=(fov_rad -fov_rad/2 + rot_rad + np.pi/2), num_points=50)
        # Rotate the sector-shaped FOV polygon based on the camera's extrinsic parameters
        verts = [[np.dot(np.eye(3), [x[i], y[i], z[i]]) + translation for i in range(len(x))]]
        fov_patch = Poly3DCollection(verts, alpha=0.2, color=(1, 0, 0))
        ax.add_collection3d(fov_patch)
        # Add camera label
        ax.text(translation[0], translation[1], translation[2], f'Camera {i+1}', color='k', fontsize=10)

    # Set the aspect ratio to be equal
    ax.set_box_aspect([1, 1, 1])
    # plt.imshow()
    plt.savefig('axis.png', bbox_inches='tight')               
    # plt.text(2, 8, '这是指定的字符串', fontsize=14)  # (x, y) 是文本的位置  

    plt.show()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='sensor')
    # please write the float, keeping 1 bit on the right of '.';
    parser.add_argument('--version', type=str)
    args = parser.parse_args()
    
    main(args)
    
   