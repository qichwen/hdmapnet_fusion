"""Camera model definitions."""

import argparse
import codecs
import json
import math

import numpy as np
from numpy.polynomial.polynomial import Polynomial
from pip import main
from scipy.optimize import curve_fit


def function(x, kc2, kc3):
    return x * (1.0 + kc2 * (x**2) + kc3 * (x**4))


def poly5_function(x, kc2, kc3, kc4, kc5):
    return x * (1.0 + kc2 * (x**2) + kc3 * (x**4) + kc4 * (x**6) + kc5 * (x**8))


def gm_polyn_function(x, kc2, kc3, kc4, kc5, kc6):
    return x * (1.0 + kc2 * (x) + kc3 * (x**2) + kc4 * (x**3) + kc5 * (x**4) + kc6 * (x**5))


def change_polyn_coeffi(origin_coeffi, fov):
    half_fov = math.radians(fov / 2.0)
    x = np.arange(-half_fov, half_fov, math.radians(1))
    origin_y = x * (
        1.0
        + origin_coeffi[0] * (x**2)
        + origin_coeffi[1] * (x**4)
        + origin_coeffi[2] * (x**6)
        + origin_coeffi[3] * (x**8)
    )
    # chang into poly3 mode
    p_est, err_est = curve_fit(function, x, origin_y)
    return (p_est[0], p_est[1], 0.0, 0.0)


def get_curve_points(coeffi, fov):
    half_fov = math.radians(fov / 2.0)
    x = np.arange(-half_fov, half_fov, math.radians(1))
    y = x * (1.0 + coeffi[0] * (x**2) + coeffi[1] * (x**4) + coeffi[2] * (x**6) + coeffi[3] * (x**8))
    return (x * 180 / math.pi, y * 180 / math.pi)


def get_gm_curve_points(coeffi, fov):
    half_fov = math.radians(fov / 2.0)
    x = np.arange(0, half_fov, math.radians(1))
    y = gm_polyn_function(x, coeffi[0], coeffi[1], coeffi[2], coeffi[3], coeffi[4])
    return (x * 180 / math.pi, y * 180 / math.pi)


def get_poly5_curve_points(coeffi, fov):
    half_fov = math.radians(fov / 2.0)
    x = np.arange(0, half_fov, math.radians(1))
    y = x * (1.0 + coeffi[0] * (x**2) + coeffi[1] * (x**4) + coeffi[2] * (x**6) + coeffi[3] * (x**8))
    return (x * 180 / math.pi, y * 180 / math.pi)


def change_gm_polyn_coeffi(origin_gm_coeffi, fov):
    half_fov = math.radians(fov / 2.0)
    x = np.arange(0, half_fov, math.radians(1))
    gm_y = x * (
        1.0
        + origin_gm_coeffi[0] * (x)
        + origin_gm_coeffi[1] * (x**2)
        + origin_gm_coeffi[2] * (x**3)
        + origin_gm_coeffi[3] * (x**4)
        + origin_gm_coeffi[4] * (x**5)
    )
    p_est, err_est = curve_fit(poly5_function, x, gm_y)
    print(err_est)
    return p_est


def change_harz_polyn_coeffi_fisheye(origin_harz_coeffi, fov):
    half_fov = math.radians(fov / 2.0)
    x = np.arange(0, half_fov, math.radians(1))
    harz_y = x * (1.0 + origin_harz_coeffi[0] * x + origin_harz_coeffi[1] * (x**2) + origin_harz_coeffi[2] * (x**3))
    p_est, err_est = curve_fit(poly5_function, x, harz_y)
    print(err_est)
    return p_est


def change_harz_polyn_coeffi_non_fisheye(origin_harz_coeffi, fov):
    half_fov = math.radians(fov / 2.0)
    x = np.arange(0, half_fov, math.radians(1))
    harz_y = x * (1.0 + origin_harz_coeffi[0] * x + origin_harz_coeffi[1] * (x**2) + origin_harz_coeffi[2] * (x**3))
    p_est, err_est = curve_fit(function, x, harz_y)
    print(err_est)
    return p_est


class IdealPinholeCamera(object):
    """Reperesents an ideal pinhole camera with no distortions."""

    def __init__(self, fov_x_deg, fov_y_deg, width, height, rolling_shutter_duration=0):
        """The __init__ function.

        Args:
            fov_x_deg (float): the horizontal FOV in degrees.
            fov_y_deg (float): the vertical FOV in degrees.
            width (int): the width of the image.
            height (int): the height of the image.
            rolling_shutter_duration (int): for rolling shutter, the time difference
                between sampling of the top and the bottom row of the camera frame.
        """
        self.width = width
        self.height = height
        self.cx = width / 2
        self.cy = height / 2
        self.center = np.asarray([self.cx, self.cy], dtype=np.float32)
        self.foc_x = None
        self.foc_y = None
        self.rolling_shutter_duration = rolling_shutter_duration
        self._focal_from_fov(fov_x_deg, fov_y_deg)
        # The intrinsics matrix
        self.k = np.asarray(
            [[self.foc_x, 0, self.cx], [0, self.foc_y, self.cy], [0, 0, 1]],
            dtype=np.float32,
        )

        # The inverse of the intrinsics matrix (for backprojection)
        self.k_inv = np.asarray(
            [
                [1.0 / self.foc_x, 0, -self.cx / self.foc_x],
                [0, 1.0 / self.foc_y, -self.cy / self.foc_y],
                [0, 0, 1],
            ],
            dtype=np.float32,
        )

    def _focal_from_fov(self, fov_x_deg, fov_y_deg):
        """Compute the focal length from horizontal and vertical FOVs.

        Args:
            fov_x_deg (float): the horizontal FOV in degrees.
            fov_y_deg (float): the vertical FOV in degrees.
        """
        fov_x = np.radians(fov_x_deg)
        fov_y = np.radians(fov_y_deg)
        self.foc_x = self.width / (2.0 * np.tan(fov_x * 0.5))
        self.foc_y = self.height / (2.0 * np.tan(fov_y * 0.5))

    def ray2pixel(self, rays):
        """Project 3D rays to 2D pixel coordinates.

        Args:
            rays (np.array): the rays.

        Returns:
            projected (np.array): the projected pixel coordinates.
            valid (np.array): the validity flag for each projected pixel.
        """
        if np.ndim(rays) == 1:
            rays = rays[np.newaxis, :]

        rays = rays.astype(np.float32)

        r = np.divide(rays, rays[:, 2:], out=np.zeros_like(rays), where=rays[:, 2:] != 0)

        projected = np.matmul(self.k, r.T).T

        # Set rays behind sensor to originate at infinity
        idx_behind = rays[:, -1] <= 0
        proj = projected[:, :2]
        proj[idx_behind] *= np.inf

        return proj

    def pixel2ray(self, pixels):
        """Backproject 2D pixels into 3D rays.

        Args:
            pixels (np.array): the pixels to backproject. Size of (n_points, 2), where the first
                column contains the `x` values, and the second column contains the `y` values.

        Returns:
            rays (np.array): the backprojected 3D rays.
        """
        if np.ndim(pixels) == 1:
            pixels = pixels[np.newaxis, :]

        pixels = pixels.astype(np.float32)

        # Add the third component of ones
        pixels = np.c_[pixels, np.ones((pixels.shape[0], 1), dtype=np.float32)]
        rays = np.matmul(self.k_inv, pixels.T).T

        # Normalize the rays
        norm = np.linalg.norm(rays, axis=1, keepdims=True)
        norm[norm == 0] = 1
        return rays / norm

    def __str__(self):
        """Returns a string representation of this object."""
        return (
            f"IdealPinholeCamera camera model:\n"
            f"center={self.center}\n\twidth={self.width}\n\theight={self.height}\n\t"
            f"foc_x={self.foc_x}\n\tfoc_y={self.foc_y}"
        )


class FThetaCamera(object):
    """Defines an FTheta camera model."""

    @staticmethod
    def from_rig(rig_file, sensor_name):
        """Helper method to initialize a new object using a rig file and the sensor's name.

        Args:
            rig_file (str): the rig file path.
            sensor_name (str): the name of the sensor.

        Returns:
            FThetaCamera: the newly created object.
        """
        with open(rig_file, "r") as fp:
            rig = json.load(fp)

        # Parse the properties from the rig file
        sensors = rig["rig"]["sensors"]
        sensor = None
        sensor_found = False

        for sensor in sensors:
            if sensor["name"] == sensor_name:
                sensor_found = True
                break

        if not sensor_found:
            raise ValueError(f"The camera '{sensor_name}' was not found in the rig!")

        return FThetaCamera.from_dict(sensor)

    @staticmethod
    def from_dict(rig_dict):
        """Helper method to initialize a new object using a dictionary of the rig.

        Args:
            rig_dict (dict): the sensor dictionary to initialize with.

        Returns:
            FThetaCamera: the newly created object.
        """
        props = rig_dict["properties"]

        if props["Model"] != "ftheta":
            raise ValueError("The given camera is not an FTheta camera")

        cx = float(props["cx"])
        cy = float(props["cy"])
        width = int(props["width"])
        height = int(props["height"])

        poly_type = None
        suppl_terms = {}

        if "bw-poly" in props.keys():
            # old ftheta format
            bw_poly = [np.float32(val) for val in props["bw-poly"].split()]
            poly_type = "pixeldistance-to-angle"
            suppl_terms["linear-c"] = 1.0
            suppl_terms["linear-d"] = 0.0
            suppl_terms["linear-e"] = 0.0

        if "polynomial" in props.keys():
            # new ftheta format
            bw_poly = [np.float32(val) for val in props["polynomial"].split()]
            assert "polynomial-type" in props.keys(), "'polynomial-type' key is missing in ftheta config!"
            poly_type = props["polynomial-type"]

            suppl_terms["linear-c"] = 1.0
            if "linear-c" in props.keys():
                suppl_terms["linear-c"] = np.float32(props["linear-c"])

            suppl_terms["linear-d"] = 0.0
            if "linear-d" in props.keys():
                suppl_terms["linear-d"] = np.float32(props["linear-d"])

            suppl_terms["linear-e"] = 0.0
            if "linear-e" in props.keys():
                suppl_terms["linear-e"] = np.float32(props["linear-e"])

        assert poly_type == "pixeldistance-to-angle", f"unsupported ftheta polynomial type: {poly_type}"
        assert (
            np.abs(suppl_terms["linear-c"] - 1.0) < 1e-5
        ), f"unsupported ftheta linear term 'linear-c': {suppl_terms['linear-c']}"
        assert (
            np.abs(suppl_terms["linear-d"]) < 1e-5
        ), f"unsupported ftheta linear term 'linear-d': {suppl_terms['linear-d']}"
        assert (
            np.abs(suppl_terms["linear-e"]) < 1e-5
        ), f"unsupported ftheta linear term 'linear-e': {suppl_terms['linear-e']}"

        return FThetaCamera(cx, cy, width, height, bw_poly)

    def __init__(self, cx, cy, width, height, bw_poly, rolling_shutter_duration=32560):
        """The __init__ method.

        Args:
            cx (float): optical center x.
            cy (float): optical center y.
            width (int): the width of the image.
            height (int): the height of the image.
            bw_poly (np.array): the backward polynomial of the FTheta model.
            rolling_shutter_duration (int): for rolling shutter, the time difference
                between sampling of the top and the bottom row of the camera frame.
        """
        self.center = np.asarray([cx, cy], dtype=np.float32)
        self.width = int(width)
        self.height = int(height)
        self.bw_poly = Polynomial(bw_poly)
        self.fw_poly, self._coeffs = self._compute_fw_poly()
        self.rolling_shutter_duration = rolling_shutter_duration

        # Other properties that need to be computed
        self._horizontal_fov = None
        self._vertical_fov = None
        self._max_angle = None
        self._max_ray_angle = None
        self._update_calibrated_camera()

    def __str__(self):
        """Returns a string representation of this object."""
        return (
            f"FTheta camera model:\n\t{self.bw_poly}\n\t"
            f"center={self.center}\n\twidth={self.width}\n\theight={self.height}\n\t"
            f"h_fov={np.degrees(self._horizontal_fov)}\n\tv_fov={np.degrees(self._vertical_fov)}"
        )

    def _update_calibrated_camera(self):
        """Updates the internals of this object after calulating various properties."""
        self._compute_fov()
        # self._max_ray_angle = max(self._horizontal_fov, self._vertical_fov) / 2
        self._max_ray_angle = (self._max_angle).copy()
        isFwPolySlopeNegativeInDomain = False
        rayAngle = (np.float32(self._max_ray_angle)).copy()
        deg2rad = np.pi / 180.0
        while rayAngle >= np.float32(0.0):
            temp_dval = self.fw_poly.deriv()(self._max_ray_angle).item()
            if temp_dval < 0:
                isFwPolySlopeNegativeInDomain = True
            rayAngle -= deg2rad * np.float32(1.0)

        if isFwPolySlopeNegativeInDomain:
            rayAngle = (np.float32(self._max_ray_angle)).copy()
            while rayAngle >= np.float32(0.0):
                rayAngle -= deg2rad * np.float32(1.0)
            raise Exception("FThetaCamera: derivative of distortion within image interior is negative")

        # Evaluate the forward polynomial at point (self._max_ray_angle, 0)
        # Also evaluate its derivative at the same point
        val = self.fw_poly(self._max_ray_angle).item()
        dval = self.fw_poly.deriv()(self._max_ray_angle).item()

        if dval < 0:
            raise Exception("FThetaCamera: derivative of distortion at edge of image is negative")

        self._max_ray_distortion = np.asarray([val, dval], dtype=np.float32)

    def _compute_fw_poly(self):
        """Computes the forward polynomial for this camera.

        This function is a replication of the logic in the following file from the DW repo:
        src/dw/calibration/cameramodel/CameraModels.cpp
        """

        def get_max_value(p0, p1):
            return np.linalg.norm(np.asarray([p0, p1], dtype=self.center.dtype) - self.center)

        max_value = 0.0

        size = (self.width, self.height)
        value = get_max_value(0.0, 0.0)
        max_value = max(max_value, value)
        value = get_max_value(0.0, size[1])
        max_value = max(max_value, value)
        value = get_max_value(size[0], 0.0)
        max_value = max(max_value, value)
        value = get_max_value(size[0], size[1])
        max_value = max(max_value, value)

        SAMPLE_COUNT = 500
        samples_x = []
        samples_b = []
        step = max_value / SAMPLE_COUNT
        x = step

        for _ in range(0, SAMPLE_COUNT):
            p = np.asarray([self.center[0] + x, self.center[1]], dtype=np.float32)
            ray, _ = self.pixel2ray(p)
            xy_norm = np.linalg.norm(ray[0, :2])
            theta = np.arctan2(float(xy_norm), float(ray[0, 2]))
            samples_x.append(theta)
            samples_b.append(float(x))
            x += step

        x = np.asarray(samples_x, dtype=np.float64)
        y = np.asarray(samples_b, dtype=np.float64)

        # Fit a 4th degree polynomial. The polynomial function is as follows:

        def f(x, b, x1, x2, x3, x4):
            """4th degree polynomial."""
            return b + x1 * x + x2 * (x**2) + x3 * (x**3) + x4 * (x**4)

        # The constant in the polynomial should be zero, so add the `bounds` condition.
        # FIXME(mmaghoumi) DW mentions disabling input normalization, what's that??
        coeffs, _ = curve_fit(
            f,
            x,
            y,
            bounds=(
                [0, -np.inf, -np.inf, -np.inf, -np.inf],
                [np.finfo(np.float64).eps, np.inf, np.inf, np.inf, np.inf],
            ),
        )
        coeffs = [np.float32(val) if i > 0 else 0 for i, val in enumerate(coeffs)]
        # Return the polynomial and hardcode the bias value to 0
        return Polynomial(coeffs), coeffs

    def pixel2ray(self, x):
        """Backproject 2D pixels into 3D rays.

        Args:
            x (np.array): the pixels to backproject. Size of (n_points, 2), where the first
                column contains the `x` values, and the second column contains the `y` values.

        Returns:
            rays (np.array): the backprojected 3D rays. Size of (n_points, 3).
            valid (np.array): bool flag indicating the validity of each backprojected pixel.
        """
        # Make sure x is n x 2
        if np.ndim(x) == 1:
            x = x[np.newaxis, :]

        # Fix the type
        x = x.astype(np.float32)
        xd = x - self.center
        xd_norm = np.linalg.norm(xd, axis=1, keepdims=True)
        alpha = self.bw_poly(xd_norm)
        sin_alpha = np.sin(alpha)

        rx = sin_alpha * xd[:, 0:1] / xd_norm
        ry = sin_alpha * xd[:, 1:] / xd_norm
        rz = np.cos(alpha)

        rays = np.hstack((rx, ry, rz))
        # special case: ray is perpendicular to image plane normal
        valid = (xd_norm > np.finfo(np.float32).eps).squeeze()
        rays[~valid, :] = (0, 0, 1)  # This is what DW sets these rays to

        # note:
        # if constant coefficient of bwPoly is non-zero,
        # the resulting ray might not be normalized.
        return rays, valid

    def ray2pixel(self, rays):
        """Project 3D rays to 2D pixel coordinates.

        Args:
            rays (np.array): the rays.

        Returns:
            result (np.array): the projected pixel coordinates.
        """
        # Make sure the input shape is (n_points, 3)
        if np.ndim(rays) == 1:
            rays = rays[np.newaxis, :]

        # Fix the type
        rays = rays.astype(np.float32)

        xy_norm = np.zeros((len(rays), 1), dtype=np.float32)
        xyz_norm = np.zeros((len(rays), 1), dtype=np.float32)

        ab = rays[:, 0] * rays[:, 0] + rays[:, 1] * rays[:, 1]
        c = rays[:, 2] * rays[:, 2]

        xy_norm[:, 0] = np.sqrt(ab)
        xyz_norm[:, 0] = np.sqrt(ab + c)

        cos_alpha = rays[:, 2:] / xyz_norm
        alpha = np.zeros_like(cos_alpha)

        cos_alpha_condition = np.logical_and(cos_alpha > np.float32(-1.0), cos_alpha < np.float32(1.0)).squeeze()
        sin_alpha_condition = np.logical_and(~cos_alpha_condition, xyz_norm.squeeze() > 0).squeeze()
        app_alpha_condition = np.logical_and(~cos_alpha_condition, ~sin_alpha_condition)

        alpha[cos_alpha_condition] = np.arccos(cos_alpha[cos_alpha_condition])
        alpha[sin_alpha_condition] = xy_norm[sin_alpha_condition] / xyz_norm[sin_alpha_condition]
        alpha[app_alpha_condition] = xy_norm[app_alpha_condition]
        delta = np.zeros_like(cos_alpha)
        alpha_cond = alpha <= self._max_ray_angle

        delta[alpha_cond] = self.fw_poly(alpha[alpha_cond])
        # For outside the model (which need to do linear extrapolation)
        delta[~alpha_cond] = (
            self._max_ray_distortion[0] + (alpha[~alpha_cond] - self._max_ray_angle) * self._max_ray_distortion[1]
        )

        # Determine the bad points with a norm of zero, and avoid division by zero
        bad_norm = xy_norm <= 0
        xy_norm[bad_norm] = 1
        delta[bad_norm] = 0
        # compute pixel relative to center
        scale = delta / xy_norm
        pixel = scale * rays

        # Handle the edge cases (ray along image plane normal)
        edge_case_cond = (xy_norm <= np.float32(0.0)).squeeze()
        pixel[edge_case_cond, :] = rays[edge_case_cond, :]
        result = pixel[:, :2] + self.center

        return result

    def _get_pixel_fov(self, pt):
        """Gets the FOV for a given point. Used internally for FOV computation.

        Args:
            pt (np.array): 2D pixel.

        Returns:
            fov (float): the FOV of the pixel.
        """
        ray, _ = self.pixel2ray(pt)
        fov = np.arctan2(np.linalg.norm(ray[:, :2]), ray[:, 2])
        return fov

    def _compute_fov(self):
        """Computes the FOV of this camera model."""
        max_x = self.width - 1
        max_y = self.height - 1

        point_left = np.asarray([0, self.center[1]], dtype=np.float32)
        point_right = np.asarray([max_x, self.center[1]], dtype=np.float32)
        point_top = np.asarray([self.center[0], 0], dtype=np.float32)
        point_bottom = np.asarray([self.center[0], max_y], dtype=np.float32)

        fov_left = self._get_pixel_fov(point_left)
        fov_right = self._get_pixel_fov(point_right)
        fov_top = self._get_pixel_fov(point_top)
        fov_bottom = self._get_pixel_fov(point_bottom)

        self._vertical_fov = fov_top + fov_bottom
        self._horizontal_fov = fov_left + fov_right
        self._compute_max_angle()

    def _compute_max_angle(self):
        """Computes the maximum ray angle for this camera."""
        max_x = self.width - 1
        max_y = self.height - 1

        p = np.asarray([[0, 0], [max_x, 0], [0, max_y], [max_x, max_y]], dtype=np.float32)

        self._max_angle = max(
            max(self._get_pixel_fov(p[0, ...]), self._get_pixel_fov(p[1, ...])),
            max(self._get_pixel_fov(p[2, ...]), self._get_pixel_fov(p[3, ...])),
        )

    def is_ray_inside_fov(self, ray):
        """Determines whether a given ray is inside the FOV of this camera.

        Args:
            ray (np.array): the 3D ray.

        Returns:
            bool: whether the ray is inside the FOV.
        """
        if np.ndim(ray) == 1:
            ray = ray[np.newaxis, :]

        ray_angle = np.arctan2(np.linalg.norm(ray[:, :2]), ray[:, 2])
        return ray_angle <= self._max_angle


def get_fisheye_para(json_input, key, fov):
    cam = FThetaCamera.from_rig(json_input, key)

    coeffi = cam._coeffs

    f = coeffi[1]
    kc2 = coeffi[2] / f
    kc3 = coeffi[3] / f
    kc4 = coeffi[4] / f
    input_coeffi = [kc2, kc3, kc4]
    new_coeffi = change_harz_polyn_coeffi_non_fisheye(input_coeffi, fov)
    fx = f
    fy = f
    cx, cy = cam.center

    camera_matrix = [float(fx), 0, float(cx), 0, float(fy), float(cy), 0, 0, 1]
    distortion_coefficients = [float(new_coeffi[0]), float(new_coeffi[1]), 0, 0]
    return camera_matrix, distortion_coefficients


def main():
    parser = argparse.ArgumentParser(
        description='convert NV ftheta camera intrinsic to opencv fisheye camera intrinsic'
    )

    optional = parser.add_argument_group('optional arguments')

    optional.add_argument(
        '--input_rig',
        type=str,
        help='input rig path, default use us_b_gls_3746.json in project',
        required=False,
        default='../../../cfg/minerva_us_calib/us_b_gls_3746.json',
    )
    optional.add_argument(
        '--input_json',
        type=str,
        help='reference json format, default use ./camParam_reference.json',
        required=False,
        default='./camParam_reference.json',
    )
    optional.add_argument(
        '--output_json',
        type=str,
        help='converted output json, default use ./camParam_out.json',
        required=False,
        default='./camParam_out.json',
    )
    args = parser.parse_args()

    key_list = [
        'camera:front:wide:120fov',
        'camera:front:tele:30fov',
        'camera:cross:left:120fov',
        'camera:cross:right:120fov',
        'camera:rear:left:70fov',
        'camera:rear:right:70fov',
        'camera:rear:tele:30fov',
    ]
    json_key = ["F120", "F30", "LF120", "RF120", "LR70", "RR70", "R30"]

    fov_list = [120, 30, 120, 120, 70, 70, 30]
    with codecs.open(args.input_json, "r", "utf-8") as f:
        meta_info = json.load(f)

    for i, (key, fov) in enumerate(zip(key_list, fov_list)):
        camera_matrix, distortion_coefficients = get_fisheye_para(args.input_rig, key, fov)
        meta_info[json_key[i]]["camera_matrix"] = camera_matrix
        meta_info[json_key[i]]["distortion_coefficients"] = distortion_coefficients

        print(key, camera_matrix, distortion_coefficients)
        # input()
    with open(args.output_json, "w", encoding='utf-8') as dump_f:
        json.dump(meta_info, dump_f, indent=4, ensure_ascii=False)

    # new_coeffi = curve_fitting.change_harz_polyn_coeffi_fisheye(input_coeffi, fov)


if __name__ == '__main__':
    main()
