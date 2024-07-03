# from moviepy.editor import VideoFileClip  
import matplotlib.pyplot as plt

import cv2  
import numpy as np  
import os
import time

import datetime

import matplotlib.pyplot as plt
from PIL import Image  
import numpy as np
import torch  

def init_tb_logger(log_dir):
    from torch.utils.tensorboard import SummaryWriter
    # from tensorboardX import SummaryWriter    
    tb_logger = SummaryWriter(log_dir=log_dir)
    return tb_logger    

def draw_tensor(sample_tokens, topdown, frame_id=0, rows=8,cols=6,imgs_path=os.path.join('plt_images', 'tempfusion'), title='tensor_draw'):
    """_summary_
    Args:
        topdown (_type_): format must be B, T, C, H, W or B, C, H, W
        frame_id (int, optional): _description_. Defaults to 0.
        rows (int, optional): _description_. Defaults to 8.
        cols (int, optional): _description_. Defaults to 6.
        imgs_path (_type_, optional): _description_. Defaults to os.path.join('plt_images', 'tempfusion').
    """
    # fig = plt.figure(figsize=(60, 40))
    # rows = 8
    # cols = 6
    imgcounter = 0
    if not os.path.exists(imgs_path):
        os.makedirs(imgs_path)
    B, T, C, H, W = topdown.shape
    if len(topdown.shape) == 4:
        fm = torch.sum(topdown.squeeze(0),0)
    elif len(topdown.shape) == 5:
        fm = torch.sum(topdown.squeeze(0).squeeze(0),0)

    # print(fm.shape)
    imgcounter += 1
    # loc = imgcounter      
    # a = fig.add_subplot(rows, cols, loc)  
    plt.imshow(fm.detach().cpu().numpy())
    
    plt.title(title)
    plt.savefig(f'{imgs_path}/{sample_tokens[0][0]}_topdown_afteripm.png', bbox_inches='tight')
    plt.close()

def visualizer_plt(x,counter, sample_token, sample_channel, ts, step, imgs_dir = ''):  
    #accept only B,C,H,W or C,H,W img!!! B*T, N, C, h, w
    
    imgs_path = os.path.join('plt_images', sample_token)
    if not os.path.exists(imgs_path):
        os.makedirs(imgs_path)
        
    if not os.path.exists(imgs_dir):
        os.makedirs(imgs_dir)
        
    if isinstance(x, Image.Image):
        # plt.axis('off')
        plt.imshow(x)
        imgname = f'plt_images/{sample_token}/{counter}_pvimg_{step}_{sample_channel}_{ts}.jpg'
        plt.savefig(imgname)
        plt.close()
        return
    elif isinstance(x, np.ndarray):
        xn = x
    elif isinstance(x, torch.Tensor):
        xn = x.detach().cpu().numpy()
    else:
        print(f"unknown type of obj while visualizer_plt_insample_{sample_token}")
        return
    
    if len(xn.shape)>5 or len(xn.shape)<=1:
        print("len(input.shape) invalid !!!")
        return
    elif len(xn.shape) == 2:
        plt.imshow(xn) #accept only (H, W)
        step = f"squeezed_multiChn_{step}"
        
    elif len(xn.shape) == 3: 
        if xn.shape[-1] == 3: #accept only (H, W, C)
            plt.imshow(xn)            
        elif xn.shape[0] == 3: #(3, H, W)
            plt.imshow(xn.transpose(1,2,0)) #accept only (H, W, C)
        elif xn.shape[0] != 3:# (C>3, H, W)
            step = f"squeezedCHW_{step}"
            xn = torch.from_numpy(xn)
            xn = torch.sum(xn,0)
            plt.imshow(xn)
            if imgs_dir != '':
                imgname = f'{imgs_dir}/{sample_token}_{counter}_{sample_channel}_{step}_{ts}.jpg'
                plt.savefig(imgname)
                plt.close()
                return               
        else:
            print("cannot display cause the dim of img")
            return         
    elif len(xn.shape) == 4: #accept only (B=1, C>3, H, W) !
        #TODO: BTCHW        
        step = f"squeezedBCHW_{step}"
        xn = torch.from_numpy(xn)
        xn = torch.sum(xn.squeeze(0),0)
        plt.imshow(xn)
        imgname = f'{imgs_dir}/{counter}_{step}_{sample_channel}_{ts}.jpg'
        plt.savefig(imgname)
        plt.close()
    # elif len(xn.shape) == 5: #B*T, N, C, h, w
    #     # e.g. (1,64,200,400),         
    #     step = f"squeezedBTCHW_{step}"
    #     xn = torch.from_numpy(xn)
    #     xn = torch.sum(xn.squeeze(0).squeeze(0),0)
    #     plt.imshow(xn) 
        
    # plt.axis('off')
    imgname = f'plt_images/{sample_token}/{counter}_{step}_{sample_channel}_{ts}.jpg'
    plt.savefig(imgname)
    plt.close()
    # plt.imshow(xn[0][:3].transpose(1,2,0)) #三通道显示

def process_videos(path, scene_num, ts_start, ts_end):
    #preconditions:
    #   1.all video shall endwith .mp4
    #   2.all timestamp files shall be store 1:1 with the video file in the same path
    #output:
    #   path : ./dataset/MBCam/input/
    #   store all the output image into certain file direction group by its camera name, 
    #   each image should be named as its camera name and the coresponding timestamp value, 
    #   into the new created file path named according to its video file name. 
    #  
    # List all files in the pointed path
    log_file_path = os.path.join(path, 'video2image.log')
    ts_start = int(float(ts_start))
    ts_end = int(float(ts_end))
    files = os.listdir(path)
    # Iterate through each file in the pointed path
    frame_range = (ts_end - ts_start) * 30 # frames in total
    for file in files:
        frame_count = 0
        frame_num = 0
        if file.endswith(".mp4"):
            video_path = os.path.join(path, file)
            time_file_path = os.path.join(path, file.replace(".mp4", ".mp4.timestamps"))
            print(f"Video processing and frames extraction for : ' {file} ' is ongoing" )
            # Read the time from the time-file
            with open(time_file_path, 'r') as time_file:
                lines = time_file.readlines()
                init_time = int(lines[0].split()[1])

            init_time = int(init_time/1000000)
            
            # Open the video file
            cap = cv2.VideoCapture(video_path)

            # Create a new directory for storing the frames #TODO image.
            new_dir_path = os.path.join(path, file.replace(".mp4", "_frames"))
            os.makedirs(new_dir_path, exist_ok=True)
            
            frame_range = (ts_end - ts_start) * 30 # frames in total from start ts
            file_num = 0
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Get the current frame's timestamp in second
                current_frame_time = init_time + (frame_num / 30)

                # Check if the current frame's timestamp is within the specified range
                if current_frame_time >= ts_start:
                    if frame_count < frame_range:
                        if frame_count == 0:
                            frame_count += 1
                            continue
                        if frame_count % 15 == 0:  # Adjust for FPS=2
                            frame_path = os.path.join(new_dir_path, f"{scene_num}_{file.replace('.mp4', '')}_{current_frame_time}_{frame_num}.jpg")
                            cv2.imwrite(frame_path, frame, [int(cv2.IMWRITE_JPEG_QUALITY), 100])     
                            file_num += 1                   
                    elif frame_count >= frame_range:
                        break                             
                    frame_num += 1
                    frame_count += 1
                else:
                    frame_num += 1
                    continue
            cap.release()

            print(f"Video processing and frames extraction for {file} completed as scene: {scene_num} . between {ts_start} till {ts_end} ")
        
        
            # 要写入的信息  
            info_to_write = f"Video processing and frames extraction for {file} completed ! as scene: {scene_num} . between {ts_start} till {ts_end}. total amount images extractted is: {file_num} "  
            
            # 打开日志文件进行写入操作  
            with open(log_file_path, 'a') as log_file:  
                # 将当前时间和信息写入日志文件  
                current_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')  
                log_file.write(f'Written images at {current_time} : \n')  
                log_file.write(info_to_write + '\n')

if __name__ == '__main__':
      
    

    # init_time is 16885 
    
    # # scene_num="n001, front_120 1:43~1:55"
    # scene_num="n001"
    # ts_start = 16988 #start ts of the scene + 103
    # ts_end = 17000 #end ts of the scene +115
    
    # # scene_num="n002, front_120 5:08~5:24"
    # scene_num="n002"
    # ts_start = 17193 #start ts of the scene, (the_past_ts_start + 308 - 103)
    # ts_end = 17209 #end ts of the scene    
    # scene_num="n002, front_120 15:48~16:12"
    
    # scene_num="n003_object"
    # ts_start = 17833 #start ts of the scene, (the_past_ts_start + 308 - 103)
    # ts_end = 17857 #end ts of the scene
    
    # v_path = "/home/qichwen/projects/HDMapNet-repo-yu/dataset/MBCam/input_test"
    # v_path = "./dataset/MBCam/input"
    # v_path = "/home/qichen/projects/Trace/pg_zone/jun-11/front30/"
    # v_path = "/home/qichen/projects/Trace/pg_zone/jun-11/front120/"
    v_path = "home/qichen/projects/Mzone_hwy/"
    
    # log_file_path = os.path.join(v_path, 'video2image.log')  
    scene_num="n005"
    ts_start = 5408 #start ts of the scene : 1:52, heading west and straight forward 24s
    ts_end = 5432 #end ts of the scene : 2:16
    
    # scene_num="n008_pgzone"
    # ts_start = 1854417.0 #start ts of the scene : 28:48, heading west and straight forward 24s #1852689 + 
    # ts_end = 1854419.0 #end ts of the scene : 28:50
    
    # scene_num="n100_pgzone"
    # ts_start = 1854407.0 #start ts of the scene : 28:38, heading west and straight forward 24s #1852689 + 
    # ts_end = 1854431.0 #end ts of the scene : 29:02
    
    process_videos(v_path, scene_num, ts_start, ts_end)

    #func - process_videos : A. A function can Loading each video data end with ".mp4" under the pointed path(given by parameters while running this .py program) with its time-file who has the same name with the video file but end with ".timestamp". B. for each video, its FPS=30, pick up all the images(the following 450 frames) started from the first unix time equals to "16970......",  the timestamp file content format for each row as below 4 rows, the first left column value is 'number of frame' , the second right column is the 'unix time value'.
    # 0	16898549946
    # 1	16898583279
    # 2	16898616612
    # 3	16898649946
    # C.store all the output image into certain file direction group by its camera name, each image should be named as its camera name and the coresponding timestamp value, into the new created file path named according to its video file name
    
#     parser = argparse.ArgumentParser(description='HDMapNet training.')
#     # logging config
#     parser.add_argument("--logdir", type=str, default='./runs')

#     # nuScenes config
#     parser.add_argument('--dataroot', type=str, default='dataset/nuScenes/')
#     parser.add_argument('--version', type=str, default='v1.0-mini', choices=['v1.0-trainval', 'v1.0-mini'])

#     # model config
#     parser.add_argument("--model", type=str, default='HDMapNet_cam')

#     # training config
#     parser.add_argument("--nepochs", type=int, default=30)
#     parser.add_argument("--max_grad_norm", type=float, default=5.0)
#     parser.add_argument("--pos_weight", type=float, default=2.13)
#     parser.add_argument("--bsz", type=int, default=4)
#     parser.add_argument("--nworkers", type=int, default=10)
#     parser.add_argument("--lr", type=float, default=1e-3)
#     parser.add_argument("--weight_decay", type=float, default=1e-7)

#     # finetune config
#     parser.add_argument('--finetune', action='store_true')
#     parser.add_argument('--modelf', type=str, default=None)

#     # data config
#     parser.add_argument("--thickness", type=int, default=5)
#     parser.add_argument("--image_size", nargs=2, type=int, default=[128, 352])
#     parser.add_argument("--xbound", nargs=3, type=float, default=[-30.0, 30.0, 0.15])
#     parser.add_argument("--ybound", nargs=3, type=float, default=[-15.0, 15.0, 0.15])
#     parser.add_argument("--zbound", nargs=3, type=float, default=[-10.0, 10.0, 20.0])
#     parser.add_argument("--dbound", nargs=3, type=float, default=[4.0, 45.0, 1.0])

#     # embedding config
#     parser.add_argument('--instance_seg', action='store_true')
#     parser.add_argument("--embedding_dim", type=int, default=16)
#     parser.add_argument("--delta_v", type=float, default=0.5)
#     parser.add_argument("--delta_d", type=float, default=3.0)

#     # direction config
#     parser.add_argument('--direction_pred', action='store_true')
#     parser.add_argument('--angle_class', type=int, default=36)

#     # loss config
#     parser.add_argument("--scale_seg", type=float, default=1.0)
#     parser.add_argument("--scale_var", type=float, default=1.0)
#     parser.add_argument("--scale_dist", type=float, default=1.0)
#     parser.add_argument("--scale_direction", type=float, default=0.2)

#     args = parser.parse_args()
#     train(args)