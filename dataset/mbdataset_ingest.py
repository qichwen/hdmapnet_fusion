# Auther: WQC

#wqc

# Load Calibrated_sensor / sensor / Scene / sample / sample_data.json files from dataset/nuScenes_mb/mb_test/, who are all composed with list of dict, please define 3 coresponding data structs for each of them for both exists and new created data ingestion in future,  format refer to below 3 examples as given records below.
# 	scene format example: [{
# 	"token": "cc8c0bf57f984915a77078b10eb33198",
# 	"log_token": "7e25a2c8ea1f41c5b0da1e69ecfa71a2",
# 	"nbr_samples": 39,
# 	"first_sample_token": "ca9a282c9e77460f8360f564131a8af5",
# 	"last_sample_token": "ed5fc18c31904f96a8f0dbb99ff069c0",
# 	"name": "scene-0061",
# 	"description": "Parked truck, construction, intersection,}]
# 	sample format example: 
# 	[{
# 	"token": "cc57c1ea80fe46a7abddfdb15654c872",
# 	"timestamp": 16990.5,
# 	"prev": "9e7683e8586542a1b6032980c45f15ce",
# 	"next": "4cf5d6c3f6ab42aab23f67b5a9782d1a",
# 	"scene_token": "fcbccedd61424f1b85dcbf8f897f9754"
# 	},]
# 	sample_data format example: 
# 	[{
# 	"token": "WToohzOUPd9NjMn7isbLXVwqrfxWYPcO",
# 	"sample_token": "DoLqFS5CTyJu0ZcDOUhV7hECOvyTymoi",
# 	"ego_pose_token": "512015c209c1490f906982c3b182c2a8",
# 	"calibrated_sensor_token": "1d31c729b073425e8e0202c5c6e66ee1",
# 	"timestamp": 16990.5,
# 	"fileformat": "jpg",
# 	"is_key_frame": true,
# 	"height": 900,
# 	"width": 1600,
# 	"filename": "/home/qichwen/projects/HDMapNet-repo-qc/dataset/nuScenes_mb/samples/CAM_FRONT/n001_camera_front_wide_120fov_16990.525314_3165.jpg",
# 	"prev": "",
# 	"next": ""
# 	},]
	
# For writting datablocks for each files after  the values on the right  are given by user via args while running this programs : 1) scene-id for example : e.g. 'n001' 2) start-timestamp and end-timestamp ( format in float e.g. 17835.0) ; 

# duplicate new data struct as sensor.json to store the mapping table between content of the loaded calibrate_sensor.json and sensor.json, for each one record of sensor whos "modality" equals to "camera", find the first record from calibrated_sensor who's value of field 'sensor_token' shares the same content with value under field "token" in sensor, then create new field 'calibrated_sensor_token' and set the value of field 'sensor_token' into the data struct sensor. 

# look up the files(count the files amount for each cam channel) of all the 6 camera channels path accordingly in input path base on scene-id,start-timestamp,end-timestamp : e.g. /dataset/nuScenes_mb/samples/CAM_FRONT/", /dataset/nuScenes_mb/samples/CAM_FRONT_LEFT/" and so on...and re-organize the data list: collecting each 6 files who share the same timestamp, means their string-timestamp in the file name all started from 17837 into 1 group.
# then for each group of data list above, reset the content of the 3 data structs as below:    
# 	1.sample_data:
#     	1.1look up the given start-timestamp and end-timestamp in the field "timestamp" of existing content in sample_data. do nothing while it could be found and exit the program and print error message, create new records base on requirements as below if none found:
# 	1.2Copy any 6 exists one record from exists content for each channel, for each one of them: 
        # TODO:
# 		1.2.1 Generate 1 same unique sample_token(32bit unique id) for field 'sample_token', 
# 		1.2.2 Generate unique sample_token(32bit unique id) for each 6 surounding camera's frame, and write into the field 'token' exist 1 above.	
#        	1.2.3 set the value for field 'calibrated_sensor_token' according to its 'timestamp' and its channel (extract channel info e.g. 'camera_front_wide', from 'filename'="/home/qichwen/projects/HDMapNet-repo-qc/dataset/nuScenes_mb/samples/CAM_FRONT/n001_camera_front_wide_120fov_16990.525314_3165.jpg") according to the updated sensor above.
#        	1.2.4 write the "timestamp" (extract timestamp info e.g. '16990.5' from 'filename'="/home/qichwen/projects/HDMapNet-repo-qc/dataset/nuScenes_mb/samples/CAM_FRONT/n001_camera_front_wide_120fov_16990.525314_3165.jpg") 
#               	1.2.5 write "filename" according to the files searching results above.
#               	1.2.6 keep the values of the rest exists fields.
#   2.sample:
# 	    2.1 Copy the value of 'sample_token', 'timestamp' from the new create sample data record as above, write it into the field "token", and "timestamp". 	
# 	  	2.2 write the timestamp
# 	  	2.3 generate new 1 token for "scene_token"
#   3.scene:
  		# 3.1 Duplcaite one record from exists content for further values setting.
  		# 3.2 write the created token above from "scene_token" of sample, into field "name"
  		# 3.2 write the scene-id into the field "name"
  		# 3.4 update the first_sample_token/last_sample_token accordingly
#             ...
# And then write the data structs content as append back to the 3 jsons with intend=4 as above and save. exit the program.

import argparse
from moviepy.editor import VideoFileClip  
import json  
import os  
from uuid import uuid4

import glob  
import sys
sys.path.append('/home/qichen/projects/HDMapNet-fusion')  
print(sys.path)
dataset_dir = 'dataset/nuScenes_mb/'
import json  
import os 
import data.const   
from data.const import CAMS_PMAP, CAMS
# import token_gen
from dataset.token_gen import *
# import copy2

def cut_videos(path, scene_num, ts_start, ts_end, log_file_path):
    
    # 指定视频文件路径  
    files = os.listdir(path)
    # Iterate through each file in the pointed path
    # frame_range = (ts_end - ts_start) * 30 # frames in total
    for file in files:
        # frame_count = 0
        # frame_num = 0
        
        if file.endswith(".mp4"):
            cut_vpath = os.path.join(path, f"cutting_videos/{file[:-4]}")
            if not os.path.exists(cut_vpath):
                os.makedirs(cut_vpath, exist_ok=True)
            video_path = os.path.join(path, file)
            time_file_path = os.path.join(path, file.replace(".mp4", ".mp4.timestamps"))
            print(f"Video processing and frames extraction for : ' {file} ' is ongoing" )
            # Read the time from the time-file
            with open(time_file_path, 'r') as time_file:
                lines = time_file.readlines()
                init_time = int(lines[0].split()[1])
            
            # init_time t0 for current video
            init_time = int(init_time/1000000)
            
            if ts_start < init_time or ts_end < init_time:
                with open(log_file_path, 'a') as log_file:  
                # 将当前时间和信息写入日志文件  
                    current_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')  
                    log_file.write(f'Invalid time offset {current_time} : \n')  
                    print(f'Invalid time offsets given s {ts_start} or e {ts_end}')
                return
            
            tss_offset = ts_start - init_time
            tse_offset = ts_end - init_time
            # 加载视频文件  
            video = VideoFileClip(video_path)  
            
            # 截取视频的指定时间段  
            clip = video.subclip(tss_offset, tse_offset)  
            
            outputv = f"{cut_vpath}/{scene_num}_{file[:-4]}_sts:{tss_offset}_ets:{tse_offset}.mp4"
            # 保存截取的视频片段  
            clip.write_videofile(outputv)
            
            #clip.write_videofile(outputv, codec="libx264")

class CalibratedSensor:  
    def __init__(self, token, sensor_token, translation, rotation, camera_intrinsic):  
        # {"token": "c81c9382edbf47af9ca8579a1ee45060",
        # "sensor_token": "1f69f87a4e175e5ba1d03e2e6d9bcd27",
        # "translation": [
        # 2.422,
        # 0.8,
        # 0.78
        # ],
        # "rotation": [
        # 0.7171539204983457,
        # 0.0,
        # 0.0,
        # 0.6969148113750004
        # ],
        # "camera_intrinsic": []
        # },
        self.token = token  
        self.sensor_token = sensor_token  
        self.translation = translation  
        self.rotation = rotation  
        self.camera_intrinsic = camera_intrinsic  
        # self.width = width  
        # self.height = height  
        # self.velocity = velocity  
        # self.name = name

    @classmethod  
    def to_dict_list(cls, listobj):  
        return [obj.__dict__ for obj in listobj] 
  
class Sensor:  
    def __init__(self, token, modality, channel, calibrated_sensor_token=''):  
        {
        "token": "ce89d4f3050b5892b33b3d328c5e82a3",
        "channel": "CAM_BACK",
        "modality": "camera"
        },
        self.token = token  
        # self.sensor_type = sensor_type  
        self.channel = channel  
        self.modality = modality  
        self.calibrated_sensor_token = calibrated_sensor_token
    
    @classmethod  
    def to_dict_list(cls, listobj):  
        return [obj.__dict__ for obj in listobj] 

        
  
class Scene:  
    def __init__(self, token, log_token, nbr_samples, first_sample_token, last_sample_token, name, description=""):  
        self.token = token  
        self.log_token = log_token  
        self.nbr_samples = nbr_samples  
        self.first_sample_token = first_sample_token  
        self.last_sample_token = last_sample_token  
        self.name = name  
        self.description = description.strip(',') if description else ""  
    
    @classmethod  
    def to_dict_list(cls, listobj):  
        return [obj.__dict__ for obj in listobj] 
    
    def copy(self):
        return Scene(self.token, self.log_token, self.nbr_samples, self.first_sample_token, self.last_sample_token, self.name,self.description)
  
class Sample:  
    def __init__(self, token, timestamp, prev, next, scene_token):  
        self.token = token  
        self.timestamp = timestamp  
        self.prev = prev  
        self.next = next  
        self.scene_token = scene_token 
         
    @classmethod  
    def to_dict_list(cls, listobj):  
        return [obj.__dict__ for obj in listobj] 
        
    def copy(self):
        return Sample(self.token, self.timestamp, self.prev, self.next, self.scene_token)
    
class SampleData:  
    def __init__(self, token, sample_token, ego_pose_token, calibrated_sensor_token, timestamp, fileformat, height, width, filename,is_key_frame=True, prev="", next=""):  
        self.token = token  
        self.sample_token = sample_token  
        self.ego_pose_token = ego_pose_token  
        self.calibrated_sensor_token = calibrated_sensor_token  
        self.timestamp = timestamp  
        self.fileformat = fileformat  
        self.is_key_frame = is_key_frame  
        self.height = height  
        self.width = width  
        self.filename = filename  
        self.prev = prev  
        self.next = next
        
    def copy(self):
        return SampleData(self.token, self.sample_token,self.ego_pose_token,self.calibrated_sensor_token,self.timestamp,self.fileformat,self.is_key_frame,self.height,self.width, self.filename,self.prev, self.next)
    
    @classmethod  
    def to_dict_list(cls, listobj):  
        return [obj.__dict__ for obj in listobj] 
    
def create_data_struct(data_dict):  
    return {  
        key: value for key, value in data_dict.items()  
    }  
  
def load_json_file(file_path):  
    with open(file_path, 'r') as file:  
        data = json.load(file)  
    return data 
 
def load_data_from_json(file_path, DataClass):  
    with open(file_path, 'r') as file:  
        data = json.load(file)        
    return [DataClass(**item) for item in data]   

def list_folders(path, prefix = 'CAM_'):  
    cam_folders = []  
    for foldername in os.listdir(path):  
        if foldername.startswith(prefix):  
            folder_path = os.path.join(path, foldername)  
            if os.path.isdir(folder_path):  
                cam_folders.append((foldername, folder_path))  
    return cam_folders

# mappings sensor - caliSensor
def create_and_save_mapping(sensor_json_path, calibrated_sensor_json_path, output_json_path):  
    sensors = sensor_json_path
    calibrated_sensors = calibrated_sensor_json_path
    
    mapping = {}  
    for sensor in sensors:  
        if sensor.modality == 'camera':  
            for calibrated_sensor in calibrated_sensors:  
                if sensor.token== calibrated_sensor.sensor_token:
                    mapping[sensor.token] = calibrated_sensor.token
                    break  # 找到匹配项后跳出循环  
      
    sensors_with_mapping = []
    sensor_copy = {}
    for sensor in sensors:  
        if sensor.modality == 'camera' and sensor.token in mapping:  
            sensor_copy['channel'] = sensor.channel  
            sensor_copy['token'] = sensor.token
            sensor_copy['modality'] = sensor.modality
            sensor_copy['calibrated_sensor_token'] = mapping[sensor.token] 
            sensor_append = sensor_copy.copy()            
            sensors_with_mapping.append(sensor_append)  
            sensor_copy.clear()
      
    # 保存新的数据结构到JSON文件  
    with open(output_json_path, 'w') as file:  
        json.dump(sensors_with_mapping, file, indent=4)  
    
    
    pass
    # 使用函数  
    
    
def ingester(sceneid, startts,endts,data_dir,output_json_path, base_path):  #args.modinput_path
    # Define data directory and file paths 
    from copy import deepcopy

    calibrated_sensor_json_path = os.path.join(data_dir, 'calibrated_sensor.json')  
    sensor_json_path = os.path.join(data_dir, 'sensor.json')  
    scene_json_path = os.path.join(data_dir, 'scene.json')  
    sample_json_path = os.path.join(data_dir, 'sample.json')  
    sample_data_json_path = os.path.join(data_dir, 'sample_data.json')  
    
    # Load data from JSON files  
    calibrated_sensors = load_data_from_json(calibrated_sensor_json_path, CalibratedSensor)  
    sensors = load_data_from_json(sensor_json_path, Sensor)  
    scenes = load_data_from_json(scene_json_path, Scene)  
    samples = load_data_from_json(sample_json_path, Sample)  
    sample_data = load_data_from_json(sample_data_json_path, SampleData)  
    # print(calibrated_sensors)
    # print(sample_data)
    
    sensor_keys = []
    for attr_name, value in sensors[-1].__dict__.items():
        sensor_keys.append(attr_name)
    #if 'calibrated_sensor_token' not in sensor_keys:    
    if not sensors[-1].calibrated_sensor_token:
        create_and_save_mapping(sensors, calibrated_sensors, output_json_path)
        sensors = load_data_from_json(output_json_path, Sensor)      
    
    # look up the files(count the files amount for each cam channel) of all the 6 camera channels path accordingly in input path 
    # base on scene-id,start-timestamp,end-timestamp : e.g. /dataset/nuScenes_mb/samples/CAM_FRONT/", 
    # /dataset/nuScenes_mb/samples/CAM_FRONT_LEFT/" and so on...and re-organize the data list: 
    # collecting each 6 files who share the same timestamp, means they share the same string-timestamp in the file name e.g. 17837 into 1 group.
    grouped_f, sc, sts, ets = find_files_and_reorg(base_path, sceneid, startts, endts)
    reorg_frames = {}
    tik = sts
    for frm in grouped_f:
        reorg_frames[str(tik)] = frm
        tik = float(tik) + 0.5
      
    
    #TODO: rework class.copy()
    print(scenes[0].__dict__.items())
    scene = 0
    
    #check scenes in exists json to see if it's new scene
    for sc in scenes:
        
        if sceneid in sc.name:
            print(f"scene existed already from scene.json :{sceneid}")            
            scene = sc.copy()
            break
        
    if scene != 0:
        # exists
        pass
    else:
        # given sceneid is new
        scene = scenes[0].copy()
        scene.token = generate_random_token()
        scene.name = sceneid
        scene.nbr_samples = len(grouped_f)
        scenes.append(scene)

    # write each json, check the returned frame's info e.g.s/e ts if already included in the loaded json
    for smpdata in sample_data:
        if 'jpg' in smpdata.fileformat and smpdata.is_key_frame == True:
            tmp_smpldata = smpdata
        if scene.name in smpdata.filename:
            if sts in smpdata.filename or ets in smpdata.filename: 
                print(f"s/e sts: {sts}, or ets:{ets}, frames already exists please check and adapt the given timestamp again, in sceneid : {scene}")
                return
    
    
    if tmp_smpldata:
        pass
    else:
        print("Chosen template sample_data.is_key_frame != True")
        return
    #Get loaded for each frame: 1.new sample 2.new sampdata     
    for frm in reorg_frames:   
        smp = samples[0].copy()        
        smp.timestamp = frm
        smp.scene_token = scene.token
        smp.token = generate_random_token()        
        for fpath in reorg_frames[frm]:
            # tmp_smpldata = sample_data[-3].copy()
            tmp_smpldata.filename = fpath
            tmp_smpldata.sample_token = smp.token
            tmp_smpldata.timestamp = smp.timestamp
            tmp_smpldata.token = generate_random_token()
            #
            for sensor in sensors:
                if sensor.channel in fpath:
                    tmp_smpldata.calibrated_sensor_token = sensor.calibrated_sensor_token
            smp_data = deepcopy(tmp_smpldata)
            sample_data.append(smp_data)   

        samples.append(smp)   
    # with open(sensor_json_path, 'w') as file:  
    #     json.dump(sensors, file, indent=4)
    #TODO: to be fixed      

    with open(scene_json_path, 'w') as file:  
        json.dump(Scene.to_dict_list(scenes), file, indent=4)
    with open(sample_json_path, 'w') as file:  
        json.dump(Sample.to_dict_list(samples), file, indent=4)
    with open(sample_data_json_path, 'w') as file:  
        json.dump(SampleData.to_dict_list(sample_data), file, indent=4)
    pass
    # calibrated_sensor_json_path = os.path.join(data_dir, 'calibrated_sensor.json')  
    # sensor_json_path = os.path.join(data_dir, 'sensor.json')  
    # scene_json_path = os.path.join(data_dir, 'scene.json')  
    # sample_json_path = os.path.join(data_dir, 'sample.json')  
    # sample_data_json_path = os.path.join(data_dir, 'sample_data.json')  
        
def find_files_and_reorg(base_path, scene_id, start_timestamp, end_timestamp):  
    
    
    file_counts = {channel: 0 for channel in CAMS}  # 初始化字典来存储每个通道的文件数量  
    grouped_files = {}  
    
    fcounter = 0
    for cam in CAMS:
        files = []
        channel_path = os.path.join(base_path, cam)  
        if os.path.exists(channel_path):
            pass
        else:
            continue
        # search_pattern = os.path.join(channel_path, f"{scene_id}+*{start_timestamp}*{end_timestamp}*")  
        ts = start_timestamp
        while int(str(ts)[:-2]) < int(str(end_timestamp)[:-2]) + 1: 
            search_pattern = os.path.join(channel_path, f"*{str(ts)[:-2]}*")
            glob_results = glob.glob(search_pattern)
            if glob_results:
                for glob_r in glob_results:
                    fcounter += 1
                    files.append(glob_r)
            ts = float(ts) + 1
        fs = files.copy()
        file_counts[cam] = fs
            
        if len(files) != fcounter:
            #TODO:
            print(f'Missing frames to be checked under: {channel_path}')
        files.clear()
    
    if fcounter > 0:
        for cam in CAMS:
            if file_counts[cam]:
                # f_c = len(file_counts[cam])
                pass
            else:
                """filling null for further zip"""
                for i in range(fcounter):
                    file_counts[cam].append('null')
    else:
        print(f'Missing frames to be checked under: {channel_path}')
        return
                
    # print(file_counts)
    # TODO: to be improved!
    framesgroup = list(zip(file_counts['CAM_FRONT_LEFT'], file_counts['CAM_FRONT'], file_counts['CAM_FRONT_RIGHT'],file_counts['CAM_BACK_LEFT'],file_counts['CAM_BACK'],file_counts['CAM_BACK_RIGHT']))
    print(framesgroup)    
    # 将文件添加到对应时间戳的列表中  
      
    return framesgroup, scene_id, start_timestamp, end_timestamp 
                    
if __name__ == '__main__':
    # input , TODO via args
    # data_dir = 'dataset/nuScenes_mb/mb_test/'  
    # scene_path = 'dataset/nuScenes_mb/mb_test/scene.json'
    # scene_data = load_json_file(scene_path)
    # create_data_struct(scene_data) 
    from image_processer import *

    from mbdataset import * 
    
    import argparse

    parser = argparse.ArgumentParser(description='mbdataset ingest')
    # please write the float, keeping 1 bit on the right of '.';
    parser.add_argument('--sceneid', type=str, default='n003')
    parser.add_argument('--startts', type=str, default='5408.0')
    parser.add_argument('--endts', type=str, default='17839.0')
    parser.add_argument('--v_path', type=str, default='17839.0')
    
    parser.add_argument('--modinput_path', type=str, default="dataset/nuscenes_mb/samples/")
    parser.add_argument('--source_path', type=str, default="/home/qichen/projects/Trace/Mzone_hwy/")
    
    parser.add_argument('--output_dir', type=str, default="dataset/nuscenes_mb/mb_test/")
    parser.add_argument('--output_json_path', type=str, default="dataset/nuscenes_mb/mb_test/sensor_mapping.json")

    args = parser.parse_args()

    # v_path = "/home/qichen/projects/Trace/Mzone_hwy/"
    #/home/qichen/projects/Mzone_hwy/
    
    # log_file_path = os.path.join(v_path, 'video2image.log')
    
    """n004_western_straight_24s"""  
    scene_num="scene-n004"  
    ts_start = .0 #start ts of the scene : 1:52, heading west and straight forward 24s
    ts_end = .0 #end ts of the scene : 2:16
    
    """n005_western_straight_24s"""  
    scene_num="scene-n005"  
    ts_start = 5408.0 #start ts of the scene : 1:52, heading west and straight forward 24s
    ts_end = 5433.0 #end ts of the scene : 2:16
    
    """eastern_ramp+straight_25s"""
        # scene-n007: "
        # ts_start = 5433.0 # sts - 2:17
        # ts_end = 5458.0 # ets - 2:41
    """pgzone_ramp+straight_25s"""

    # scene_num="n100_pgzone"
    # ts_start = 1854407.0 #start ts of the scene : 28:38, heading west and straight forward 24s #1852689 + 
    # ts_end = 1854431.0 #end ts of the scene : 29:02
    
    videopath = "/home/qichen/projects/Trace/Mzone_hwy/"
    log_file_path = "/home/qichen/projects/Trace/Mzone_hwy/logs"
    
    """raw video cut"""
    # cut_videos(videopath, scene_num, ts_start, ts_end, log_file_path)
    
    """frame extracter"""
    # process_videos(args.v_path, args.sceneid, args.startts, args.endts) 
    # input 
    
    """frame pre-processer (distortion) and transit to sample/"""
    """CAMS_PMAP = {
    'CAM_FRONT_LEFT': 'camera_cross_left_120fov_frames',
    'CAM_FRONT':'camera_front_wide_120fov_frames',
     # 'CAM_FRONT':'camera_front_tele_30fov_frames',
    'CAM_FRONT_RIGHT':'camera_cross_right_120fov_frames',
    'CAM_BACK_LEFT':'camera_rear_left_70fov_frames',
    'CAM_BACK':'camera_rear_tele_30fov_frames',
    'CAM_BACK_RIGHT':'camera_rear_right_70fov_frames',
    }
    """
    imgs_preprocessor(args.modinput_path, args.source_path, args.sceneid, crop=False)
    # print(cam_maps) 
    
    ingester(args.sceneid, args.startts, args.endts, args.output_dir, args.output_json_path, args.modinput_path)
    
