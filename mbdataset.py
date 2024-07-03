import os
import numpy as np

import torch
import torchvision
from PIL import Image
from pyquaternion import Quaternion
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
from data import const
import json
import shutil

class MyDataset(Dataset):
    #1. #del all img from path/cam, then trans imgs from source path, img crop 
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
            
        for filename in os.listdir(self.path):  
            if filename.endswith(".jpg"):  
                f = os.path.join(self.path, filename)
                os.remove(f)
                 
        for filename in os.listdir(self.source_path):  
            if (self.sceneid in filename) and f'{const.CAMS_PMAP[self.channel][:-8]}' in filename and filename.endswith(".jpg") :  
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
        
def imgs_preprocessor(modinput_path, source_path, scenen, crop):
    # read get_mean_std for each channel
    # return a dict{} incl. channel - mean / std
    #modinput_path = 'dataset/nuscenes_mb/samples/'
    
    #cams = const.CAMS
    cams_map = {}
    # all_chn=os.listdir(modinput_path)
    
    folders_list = list_folders(modinput_path)  
    # rawimg_flist = list_folders(source_path, 'camera_')
    # print(rawimg_flist)
    
    # for folder_names, folder_path,  in folders_list:  
    #     print("Folder Path:", folder_path)  
    #     print("Folder Names:", folder_names)
        
    for folder_name, folder_path  in folders_list:
        if os.path.exists(os.path.join(source_path, const.CAMS_PMAP[folder_name])):
            #for each channel:  
            dataset=MyDataset(folder_path, folder_name, os.path.join(source_path, const.CAMS_PMAP[folder_name]), crop, scenen)
            dataloader=torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)
            mean, std = get_mean_std(dataloader)
            cams_map[folder_name] = (mean, std)
        else:
            continue
    
    #write them into json file, #would not being used since 6-3-2024
    with open(f"cams_map.json", "w") as file:  
        json.dump(cams_map, file, indent=4)
    pass

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