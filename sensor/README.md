# Ftheta to OpenCV fisheye camera intrinsic tool

In Gen6 DC fleets, they use a camera model called Ftheta, which is not very common use. This tool read the rig file of Gen6 dev fleet and convert it camera intrinsic parameters from ftheta to OpenCV camera model intrinsic(see more details in: https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html )


## Usage

### Example

```bash
python3 ftheta_to_fisheye.py --input_rig ../../../cfg/minerva_us_calib/us_b_gls_3746.json --input_json ./camParam_reference.json --output_json ./camParam_out.json
```

### Parameters

```bash
usage: ftheta_to_fisheye.py [-h] [--input_rig INPUT_RIG] [--input_json INPUT_JSON] [--output_json OUTPUT_JSON]

convert NV ftheta camera intrinsic to opencv fisheye camera intrinsic

optional arguments:
  -h, --help            show this help message and exit

optional arguments:
  --input_rig INPUT_RIG
                        input rig path, default use us_b_gls_3746.json in project
  --input_json INPUT_JSON
                        reference json format, default use ./camParam_reference.json
  --output_json OUTPUT_JSON
                        converted output json, default use ./camParam_out.json
```

