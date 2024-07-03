import cv2
import numpy as np
import os
import argparse
import numpy as np
from PIL import Image
import geopandas as gpd
from shapely.geometry import Point
import folium
from sqlalchemy import create_engine


def extract_frames(video_path, interval=1):
    cap = cv2.VideoCapture(video_path)
    frames = []
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % interval == 0:
            frames.append(frame)
        frame_count += 1

    cap.release()
    return frames

def detect_and_match_features(frames):
    orb = cv2.ORB_create()
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    kp_list, des_list = [], []

    for frame in frames:
        kp, des = orb.detectAndCompute(frame, None)
        kp_list.append(kp)
        des_list.append(des)

    matches = []
    for i in range(1, len(frames)):
        matches.append(bf.match(des_list[i-1], des_list[i]))

    return kp_list, matches

def estimate_camera_pose(kp_list, matches):
    poses = []
    pose = np.eye(4)  # Initial pose (identity matrix)
    poses.append(pose)

    for i in range(1, len(matches)):
        src_pts = np.float32([kp_list[i-1][m.queryIdx].pt for m in matches[i-1]]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp_list[i][m.trainIdx].pt for m in matches[i-1]]).reshape(-1, 1, 2)
        E, mask = cv2.findEssentialMat(src_pts, dst_pts, focal=1.0, pp=(0, 0), method=cv2.RANSAC, prob=0.999, threshold=1.0)
        _, R, t, mask = cv2.recoverPose(E, src_pts, dst_pts)

        # Update pose
        new_pose = np.eye(4)
        new_pose[:3, :3] = R
        new_pose[:3, 3] = t.flatten()
        pose = pose @ new_pose
        poses.append(pose)

    return poses

def georeference_poses(poses, hd_map):
    # Assuming hd_map is a GeoDataFrame with WGS84 coordinates
    trajectory = []
    for pose in poses:
        x, y, z = pose[:3, 3] # [0,3], [1,3], [2,3]
        point = Point(x, y)  # Simplified for illustration; actual implementation may vary
        trajectory.append(point)

    trajectory_gdf = gpd.GeoDataFrame(geometry=trajectory, crs=hd_map.crs)
    return trajectory_gdf

def visualize_trajectory(trajectory_gdf):
    center = [trajectory_gdf.geometry.y.mean(), trajectory_gdf.geometry.x.mean()]
    m = folium.Map

def load_hd_map_from_postgres(db_connection_string):
    engine = create_engine(db_connection_string)
    overall_geometry = gpd.read_postgis("SELECT * FROM test_0730_mzone.lane_boundary", engine, geom_col='geometry')
    lane_divider = gpd.read_postgis("SELECT * FROM lane_divider", engine, geom_col='geometry')
    pedestrian_crossing = gpd.read_postgis("SELECT * FROM pedestrian_crossing", engine, geom_col='geometry')
    contour = gpd.read_postgis("SELECT * FROM contour", engine, geom_col='geometry')
    return overall_geometry, lane_divider, pedestrian_crossing, contour

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # logging config
    parser.add_argument("--video_path", type=str, default='./VO')
    parser.add_argument("--map_path", type=str, default='./VO')

    # Connection with map
    db_connection_string = 'postgresql://postgres:123@127.20.10.6:5432/mz'
    overall_geometry, lane_divider, pedestrian_crossing, contour = load_hd_map_from_postgres(db_connection_string)

    video_path = 'VO/F30_n001.mp4'
    frames = extract_frames(video_path, interval=15)  # Extract one frame every second
    kp_list, matches = detect_and_match_features(frames)
    poses = estimate_camera_pose(kp_list, matches)

    # Example HD map (replace with your actual map data)
    hd_map = gpd.read_file('path/to/hd_map.geojson')

    trajectory_gdf = georeference_poses(poses, hd_map)

    visualize_trajectory(trajectory_gdf)
