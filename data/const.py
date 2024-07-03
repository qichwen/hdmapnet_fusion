MAP = ['boston-seaport', 'singapore-hollandvillage', 'singapore-onenorth', 'singapore-queenstown']
CAMS = ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT']

CLASS2LABEL = {
    'road_divider': 0,
    'lane_divider': 0,
    'ped_crossing': 1,
    'contours': 2,
    'others': -1
}

NUM_CLASSES = 3
IMG_ORIGIN_H = 900
IMG_ORIGIN_W = 1600

CAMS_PMAP = {
    'CAM_FRONT_LEFT': 'camera_cross_left_120fov_frames',
    # 'CAM_FRONT':'camera_front_wide_120fov_frames',
    'CAM_FRONT':'camera_front_tele_30fov_frames',
    'CAM_FRONT_RIGHT':'camera_cross_right_120fov_frames',
    'CAM_BACK_LEFT':'camera_rear_left_70fov_frames',
    'CAM_BACK':'camera_rear_tele_30fov_frames',
    'CAM_BACK_RIGHT':'camera_rear_right_70fov_frames',
}

