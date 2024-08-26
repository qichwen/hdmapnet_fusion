import os
import numpy as np

import torch
import torchvision
from PIL import Image
from pyquaternion import Quaternion
import glob
#mb-test
# from nuscenes import NuScenes
from nuscenes.nuscenes import NuScenes
#mb-test
from nuscenes.utils.splits import create_splits_scenes

from torch.utils.data import Dataset
from data.rasterize import preprocess_map
# from .const import CAMS, NUM_CLASSES, IMG_ORIGIN_H, IMG_ORIGIN_W
# from .vector_map import VectorizedLocalMap
# from .lidar import get_lidar_data
# from .image import normalize_img, img_transform
# from .utils import label_onehot_encoding
from model.voxel import pad_or_trim_to_np
from image_processer import visualizer_plt
from data.const import *
import json
import shutil
from cv_distortion import *

class MyDataset(Dataset):
    #1. #del all img from each cam folder under given modinput_path, then trans imgs from source path, img crop is optional 
    #2. Calulating and storage the mean and std for current channel_modality
    #3. *DONE : reply current channel mean and std to Normalization method under dataset.py
        
    def __init__(self,path='dataset/nuscenes_mb/samples/', cam_name='', source_path = '',__crop__=False, scenen=''):
        super().__init__()
        # self.source_path = "dataset/MBCam/input/camera_front_tele_30fov_frames/"
        self.path=path
        self.crop = __crop__
        self.source_path = source_path
        self.channel=cam_name
        all_imgs=os.listdir(path)#获取全部图片的名字
        self.imgs = []
        self.sceneid = scenen
        #将每张图片的所在路径读取进来，保存在self.imgs中
        #del all img from samples/, then trans imgs from source path    
        self.restore_ori_crop()             
    
    def restore_ori_crop(self):
        
        if not os.path.exists(self.path):  
            # os.makedirs(self.path)
            print("no image input:" + self.path)
            return 
        if not os.path.exists(self.source_path):  
            print("no image source raw input path: " + self.source_path)
            return     
        
        """clear up all historical imgs under target path"""
        for filename in os.listdir(self.path):  
            if filename.endswith(".jpg"):  
                f = os.path.join(self.path, filename)                
                os.remove(f)
                 
        for filename in os.listdir(self.source_path):  
            if (self.sceneid in filename) and f'{CAMS_PMAP[self.channel][:-8]}' in filename and filename.endswith(".jpg") :  
                source_file = os.path.join(self.source_path, filename)  
                target_file = os.path.join(self.path, filename)  
                
                # 复制文件  
                shutil.copy2(source_file, target_file)  
                print(f"Copied {source_file} to {target_file}")
                self.imgs.append((target_file, filename))
                #TODO: crop the only front!!! be careful!                
                if self.crop:
                    self.img_crop(Image.open(target_file).convert('RGB'), target_file)     
        
    def __len__(self):
        return len(self.imgs)


    def __getitem__(self, index):
        # pop base on given scene
        # print(image.size)
        image = self.imgs[index][0]
        image=Image.open(image).convert('RGB')#读进去是RGBA，需要转换一下
        image=np.array(image)#将PIL图像转成数值
        image=np.array(image).astype(np.float32).transpose((2, 0, 1))#image是HWC的，这里转为CHW
        return image, self.channel
        
    def img_crop(self, img, image_path):
        #crop & rewrite the img
        print(img.size)
        # For only mb front 0, 1500, 3840, 1836 3840-2160 !!!!
        # img_ = img.crop((0, 1380, 3840, 1836))
        img_ = img.crop((0, 1500, 3840, 1836))
        # img.resize((352,128),Image.BICUBIC)   
        print(img_.size)
        _img = img_.resize((3840,2160))
        print(_img.size)
        imgname = f'{image_path}'
        print(imgname)
        if os.path.exists(imgname):
            os.remove(imgname)
        _img.save(imgname)

def distort_images(input_folder, output_folder, camParam, scene, resolution=(3840, 2160)):
    """批量去除图像畸变
    :param input_folder: 输入图像文件夹的路径
    :param output_folder: 输出图像文件夹的路径
    :param K: 相机内参，np.array(3x3)
    :param D: 相机镜头畸变系数，np.array(4x1)
    :param resolution: 图像分辨率
    """
    K = np.array(camParam["camera_matrix"], dtype=np.float32).reshape((3, 3))
    D = np.array(camParam["distortion_coefficients"], dtype=np.float32)
    # 获取所有图像文件的路径
    images_path = glob.glob(os.path.join(input_folder, '*'))
    
    for img_path in images_path:
        img_name = os.path.basename(img_path)
        if not img_name.startswith(scene):
            continue
        output_path = os.path.join(output_folder, img_name)
        
        # 使用OpenCV读取图像
        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            print(f"Failed to read image: {img_path}")
            continue
        # 调整图像大小
        img_bgr = cv2.resize(img_bgr, resolution)
        # 去畸变
        img_distort = cv2.fisheye.undistortImage(img_bgr, K, D, None, K, resolution)
        # 保存图像
        cv2.imwrite(output_path, img_distort, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
        print(f"Processing image: {img_path}")
        print(f"Image {img_name} processed and saved!")

def get_mean_std(loader):
    channels_sum,channels_squared_sum,num_batches=0,0,0
    #这里的data就是image,shape为[batchsize,C,H,W]
    for data, _ in loader:
        channels_sum+=torch.mean(data.float().div(255),dim=[0,2,3])
        #print(channels_sum.shape)#torch.Size([3])
        channels_squared_sum+=torch.mean(data.float().div(255)**2,dim=[0,2,3])
        #print(channels_squared_sum.shape)#torch.Size([3])
        num_batches+=1
    
    #计算E(X),这也就是要求的均值
    e_x=channels_sum/num_batches
    #计算E(X^2)
    e_x_squared=channels_squared_sum/num_batches
    
    var=e_x_squared-e_x**2
    
    return e_x.tolist() ,(var**0.5).tolist()

def list_folders(path):  
    folders = []  
    for root, dirs, files in os.walk(path):  
        if dirs:  
            folders.append((root, dirs))  
    return folders  

def list_folders(path, prefix = 'CAM_'):  
    cam_folders = []  
    for foldername in os.listdir(path):  
        if foldername.startswith(prefix):  
            folder_path = os.path.join(path, foldername)  
            if os.path.isdir(folder_path):  
                cam_folders.append((foldername, folder_path))  
    return cam_folders     
        
def imgs_preprocessor(modinput_path, source_path, scene, crop):
    # read get_mean_std for each channel 
    # distortion on each img under source path
    # return a dict{} incl. channel - mean / std
    # modinput_path = 'dataset/nuscenes_mb/samples/'
    
    cams_map = {}
    
    folders_list = list_folders(modinput_path)
        
    for folder_name, folder_path  in folders_list:
        if os.path.exists(os.path.join(source_path, CAMS_PMAP[folder_name])):
            #for each channel:  
            dataset=MyDataset(folder_path, folder_name, os.path.join(source_path, CAMS_PMAP[folder_name]), crop, scene)
            dataloader=torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)            
            """values below not used so far, since we'll use the orignal nuscene's one"""
            # mean, std = get_mean_std(dataloader)
            # cams_map[folder_name] = (mean, std)
        else:
            continue 
    
    """CAMS_PMAP = {
    'CAM_FRONT_LEFT': 'camera_cross_left_120fov_frames',
    #'CAM_FRONT':'camera_front_wide_120fov_frames',
    'CAM_FRONT':'camera_front_tele_30fov_frames',
    'CAM_FRONT_RIGHT':'camera_cross_right_120fov_frames',
    'CAM_BACK_LEFT':'camera_rear_left_70fov_frames',
    'CAM_BACK':'camera_rear_tele_30fov_frames',
    'CAM_BACK_RIGHT':'camera_rear_right_70fov_frames',
    }

    CAMPARAM_MAP = {
        'LF120': 'camera_cross_left_120fov_frames',
        'F30': 'camera_front_tele_30fov_frames',
        #'F120': 'camera_front_wide_120fov_frames',
        'RF120': 'camera_cross_right_120fov_frames',
        'LR70': 'camera_rear_left_70fov_frames',
        'R30': 'camera_rear_tele_30fov_frames',
        'RR70': 'camera_rear_right_70fov_frames',
    }
    """
    # 基于文件夹名创建从CAMPARAM_MAP到CAMS_PMAP的反向映射
    folder_to_cam_map = {v: k for k, v in CAMS_PMAP.items()}

    for folder_name, folder_path  in folders_list:  # DO NOT MOVE!
        dataset=MyDataset(folder_path, folder_name, os.path.join(source_path, CAMS_PMAP[folder_name]), crop, scene)
        dataloader=torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)
        mean, std = get_mean_std(dataloader)
        cams_map[folder_name] = (mean, std)
    
    camParams = "sensor/camParam_out.json"
    with open(camParams, "r") as file:  
        camParam = json.load(file)
        
    for cam_name, folder_name in CAMPARAM_MAP.items():
        if cam_name in camParam:
            camera_params = camParam[cam_name]
            input_folder = os.path.join(source_path, folder_name)
            # 使用反向映射找到对应的CAMS_PMAP键
            output_cam_name = folder_to_cam_map.get(folder_name)
            # 确保找到了对应的输出文件夹名
            if output_cam_name:
                output_folder = os.path.join(modinput_path, output_cam_name)
                if not os.path.exists(output_folder):
                    os.makedirs(output_folder)
                distort_images(input_folder, output_folder, camera_params, scene)
                print("Distort successfully!")
            else:
                print(f"未找到与'{cam_name}'对应的输出文件夹名。")
    #write them into json file, #would not being used since 6-3-2024
    # with open(f"cams_map.json", "w") as file:  
    #     json.dump(cams_map, file, indent=4)       

if __name__ == '__main__':
    #1. pre-process each img under each channel
    #2. calculate and store all the mean/std for each channel cams_map.json for further usage
    
    modinput_path = 'dataset/nuscenes_mb/samples/'
    source_path = "/home/qichen/projects/Trace/pg_zone/jun-11/front120/"
    scenen = 'n100_pgzone'
    #TODO: pointed scene for norm params each chn
    
    cam_maps= imgs_preprocessor(modinput_path, source_path, scenen, crop = True)
    print(cam_maps)

    # dataset=MyDataset(image_pth)
 
    # #dataset/nuScenes_mb/samples/
    # dataloader=torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)
    # mean, std = get_mean_std(dataloader)
    # #deal single channel images at once.    
    
    # print(mean)
    # print(std)