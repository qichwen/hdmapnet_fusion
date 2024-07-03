import cv2
from PIL import Image
import numpy as np


cam_intrinsics = np.array([[1872.2064208984375, 0.0, 1912.15283203125],
                           [0.0, 1872.2064208984375, 1501.535888671875],
                           [0.0, 0.0, 1.0]], dtype=np.float32)

distortion = np.array([[0.0018148268052000864],
                       [-0.018532995988586563],
                       [0.0],
                       [0.0]], dtype=np.float32)


def distort(
    img_before='dataset/MBCam/input/camera_cross_left_120fov_frames/n002_camera_cross_left_120fov_17198.5_9015.jpg',
    img_after='sensor/img_after_orisize.jpg',
    K=cam_intrinsics,
    D=distortion,
    resolution=(3840, 2160)
):
    """使用 OpenCV 图像去畸变

    :param img_before: 要处理的图像
    :param img_after: 处理后的图像完整路径
    :param K: 相机内参，np.array(3x3)
    :param D: 相机镜头畸变系数，np.array(4x1)
    :param resolution: 图像分辨率

    """

    img = Image.open(img_before)
    img = img.resize(resolution)
    img_bgr = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)

    img_distort = cv2.fisheye.undistortImage(
        img_bgr,
        K,
        D,
        None,
        K,
        resolution
    )
    img_distort = cv2.cvtColor(img_distort, cv2.COLOR_BGR2RGB)
    img_distort = Image.fromarray(img_distort)
    # img_distort = img_distort.resize((1920, 1080))
    img_distort.save(img_after)


def main():
    distort()

if __name__ == "__main__":  
    main()