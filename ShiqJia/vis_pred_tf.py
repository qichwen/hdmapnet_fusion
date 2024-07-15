import os
import argparse
import numpy as np
from PIL import Image
import gc

import matplotlib.pyplot as plt
import cv2
import tqdm
import torch
import shutil

from data.dataset import semantic_dataset
from data.const import NUM_CLASSES
from model import get_model
from postprocess.vectorize import vectorize
from data.dataset_TF import *


def onehot_encoding(logits, dim=1):
    max_idx = torch.argmax(logits, dim, keepdim=True)
    one_hot = logits.new_full(logits.shape, 0)
    one_hot.scatter_(dim, max_idx, 1)
    return one_hot

def vis_resutls(args, img_name, batchi, sample_token_str):
    batchi = 0
    # cluster imge path
    img_path = f'plt_images/{sample_token_str}/'
    # cluster_path = os.path.join(img_path, 'image_cluster')
    if not os.path.exists(img_path):
        os.mkdir(img_path)
    smaple_token = sample_token_str
    # samples = read_json_file(args.image_json)
    # for smaple_token in samples:
    #TODO: to add try exception logic
    BEV_topdown = [f for f in os.listdir(img_path) if (f.startswith('topdown_afterIPM_Usamp_')) and f.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))][0]  
    BEV_topdown = os.path.join(img_path, BEV_topdown)
    
    segment_img = [f for f in os.listdir(img_path) if (f.startswith('segmentation_')) and f.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))][0]  
    segment_img = os.path.join(img_path, segment_img)

    vector_img = [f for f in os.listdir(img_path) if (f.startswith('eval_')) and f.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))][0]  
    vector_img = os.path.join(img_path, vector_img)

    overall_img = [f for f in os.listdir(img_path) if (f.startswith('6viewsfused')) and f.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))][0]  
    overall_img = os.path.join(img_path, overall_img)
   
    seg_bev_img_ori = [f for f in os.listdir(img_path) if (f.startswith('fm_overall_')) and f.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))][0]  
    seg_bev_img = os.path.join(img_path, seg_bev_img_ori)
    
    seg_bev_ori = Image.open(seg_bev_img)
    seg_bev = seg_bev_ori.resize((2200, 1200))
    
    bev_img_cam_ori = Image.open(BEV_topdown)
    bev_img_cam = bev_img_cam_ori.resize((2200, 1200))
    
    # plt.imshow(bev_img_cam)
    # plt.axis('off')
    # plt.savefig(f'BEV_topdown_.jpg')
    seg_img_cam_ori = Image.open(segment_img)
    seg_img_cam = seg_img_cam_ori.resize((2200, 1200))
    # plt.imshow(seg_img_cam)
    # plt.axis('off')
    # plt.savefig(f'seg_.jpg')
    vec_img_cam_ori = Image.open(vector_img)
    vec_img_cam = vec_img_cam_ori.resize((1600, 1200))
    # plt.imshow(vec_img_cam)
    # plt.axis('off')
    # plt.savefig(f'vec_.jpg')
    overall_img_ori = Image.open(overall_img)
    overall_img_cam = overall_img_ori.resize((6000, 3600))            
    # if not os.path.exists(segment_sample_path):
    #     os.mkdir(segment_sample_path)
    # shutil.copy(segment_img, segment_sample_path)
    #
    new_cam_image = Image.new('RGB', (10000, 5000))
    ###pastes the overall_img image onto the new_cam_image at the specified position (0, 1800). This means that 
    # the top-left corner of overall_img will be placed at the x-coordinate 0 and the y-coordinate 1800 on the new_cam_image
    new_cam_image.paste(overall_img_cam, (0, 1200+1200*2))
    new_cam_image.paste(bev_img_cam, (0, 0))
    new_cam_image.paste(seg_bev, (2200, 0))
    new_cam_image.paste(seg_img_cam, (4400, 0))
    new_cam_image.paste(vec_img_cam, (6600, 0))
    
    raw_imgs = sorted([f for f in os.listdir(img_path) if (((('fm_imgafter_norm') in f) or ('pvimg_before_norm' in f)) and f.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')))])  
    for imgn, image_file in enumerate(raw_imgs):
        rows = imgn//6
        rols = imgn%6  
        image_path = os.path.join(img_path, image_file)                  
        pv = Image.open(image_path).resize((1600,1200)) 
        new_cam_image.paste(pv, (rols * 1600, 1200+1200*rows))      

    img_dir = f'cluster_result'
    if not os.path.exists(img_dir):
        os.mkdir(img_dir)
    
    img_name = f'cluster_{batchi:06}_{smaple_token}.jpg'    
    new_image_path = os.path.join(img_dir, img_name)   
    new_cam_image.save(new_image_path)
    shutil.copy(new_image_path, img_path)
    plt.close()
    # batchi = batchi + 1    
    print(f"exported to {new_image_path}")
    
def vis_segmentation(model, val_loader):
    model.eval()
    with torch.no_grad():
        segment_path = os.path.join(os.getcwd(), 'segment_result')
        

        if not os.path.exists(segment_path):
            os.mkdir(segment_path)
        for batchi, (imgs, trans, rots, intrins, post_trans, post_rots, car_trans, yaw_pitch_roll, segmentation_gt, instance_gt, direction_gt, sample_token) in enumerate(val_loader):
            if torch.cuda.is_available():
                semantic, embedding, direction = model(imgs.cuda(), trans.cuda(), rots.cuda(), intrins.cuda(),
                                                    post_trans.cuda(), post_rots.cuda(), car_trans.cuda(), yaw_pitch_roll.cuda(), sample_token)
                semantic = semantic.softmax(1).cpu().numpy()
            else:
                semantic, embedding, direction = model(imgs, trans, rots, intrins,
                                                    post_trans, post_rots, lidar_data,
                                                    lidar_mask, car_trans, yaw_pitch_roll,sample_token[0])
                semantic = semantic.softmax(1).numpy()
            # compare with gt, set NaN to non-valid value? #TODO
            # print(semantic_gt.shape)
            # print(semantic.shape)
            # print("======shape of gt, then semantic as above======")

            # print(semantic_gt)
            # print("======semantic_gt above======")
            # print(semantic_gt < 0.1)
            # print("======semantic_gt < 0.1 above======")
            # print(semantic[semantic_gt < 0.1])
            # print("======output semantic_gt < 0.1 above======")
            # print(semantic_gt[0][0][199])
            #semantic[semantic_gt < 0.1] = np.nan
            sample_token_str = sample_token[0][0]
            pltimage_dir = os.path.join('plt_images', sample_token_str)
            if not os.path.exists(pltimage_dir):
                os.mkdir(pltimage_dir)
            for si in range(semantic.shape[0]):
                #print(len(semantic[0][1]))
                #print(semantic[0][1])
                plt.figure(figsize=(4, 2), dpi=400)
                
                plt.imshow(semantic[si][0], alpha=0.6)
                plt.xlim(0, 400)
                plt.ylim(0, 200)
                #plt.axis('off')
                imname = f'segment_[0][0]_other_{sample_token_str}.jpg'
                image_path = os.path.join(pltimage_dir, imname)                
                print('saving', image_path)                
                plt.savefig(image_path)
                plt.close()
                
                plt.imshow(semantic[si][1], vmin=0, cmap='Blues', vmax=1, alpha=0.6)
                plt.xlim(0, 400)
                plt.ylim(0, 200)
                #plt.axis('off')
                imname = f'segment_[0][1]_blues_{sample_token_str}.jpg'
                image_path = os.path.join(pltimage_dir, imname)                
                print('saving', image_path)
                
                plt.savefig(image_path)
                plt.close()
                
                plt.imshow(semantic[si][2], vmin=0, cmap='Reds', vmax=1, alpha=0.6)
                plt.xlim(0, 400)
                plt.ylim(0, 200)
                #plt.axis('off')
                imname = f'segment_[0][2]_reds_{sample_token_str}.jpg'
                image_path = os.path.join(pltimage_dir, imname)                
                print('saving', image_path)
                plt.savefig(image_path)
                plt.close()
                
                plt.imshow(semantic[si][3], vmin=0, cmap='Greens', vmax=1, alpha=1)
                plt.xlim(0, 400)
                plt.ylim(0, 200)
                plt.axis('off')
                imname = f'segment_[0][3]_greens_{sample_token_str}.jpg'
                image_path = os.path.join(pltimage_dir, imname)                
                print('saving', image_path)
                plt.savefig(image_path)
                plt.close()

                # #seg gt overall geometry
                # plt.imshow(semantic_gt[si][0],alpha=1)
                # plt.xlim(0, 400)
                # plt.ylim(0, 200)
                # plt.axis('off')
                # imname = f'segmentGT_[0][0]_{sample_token_str}.jpg'
                # image_path = os.path.join(pltimage_dir, imname)                
                # print('saving', image_path)
                # plt.savefig(image_path)
                # plt.close()
                
                # # road / lane divider
                # plt.imshow(semantic_gt[si][1], alpha=1)
                # plt.xlim(0, 400)
                # plt.ylim(0, 200)
                # plt.axis('off')
                # imname = f'segmentGT_[0][1]_{sample_token_str}.jpg'
                # image_path = os.path.join(pltimage_dir, imname)                
                # print('saving', image_path)
                # plt.savefig(image_path)
                # plt.close()
                
                # # pedestrain crossing
                # plt.imshow(semantic_gt[si][2], alpha=1)
                # plt.xlim(0, 400)
                # plt.ylim(0, 200)
                # plt.axis('off')
                # imname = f'segmentGT_[0][2]_{sample_token_str}.jpg'
                # image_path = os.path.join(pltimage_dir, imname)                
                # print('saving', image_path)
                # plt.savefig(image_path)
                # plt.close()
                
                # # contour
                # plt.imshow(semantic_gt[si][3], alpha=1)
                # plt.xlim(0, 400)
                # plt.ylim(0, 200)
                # plt.axis('off')
                # imname = f'segmentGT_[0][3]_{sample_token_str}.jpg'
                # image_path = os.path.join(pltimage_dir, imname)                
                # print('saving', image_path)
                # plt.savefig(image_path)
                # plt.close()
                
                # fig.axes.get_xaxis().set_visible(False)
                # fig.axes.get_yaxis().set_visible(False)
                plt.xlim(0, 400)
                plt.ylim(0, 200)
                #plt.axis('off')
                #TODO: print seg gt
                imname = f'eval_segment_{batchi:06}_{sample_token_str}_{si:03}.jpg'
                image_path = os.path.join(segment_path, imname)
                print('saving', image_path)
                plt.savefig(image_path)
                
                #print(os.path.exists(pltimage_dir))
                
                pltimage_path = f'{pltimage_dir}/segmentation_{batchi:06}_{sample_token_str}_{si:03}.jpg'
                plt.savefig(pltimage_path)
                
                # fig.clf()

                plt.close()                
                gc.collect()
            # del semantic
            gc.collect()
    


def vis_vector(model, val_loader, angle_class):
    model.eval()
    car_img_path = os.path.join(os.getcwd(), 'icon', 'car.png')
    car_img = Image.open(car_img_path)

    with torch.no_grad():
        vector_path = os.path.join(os.getcwd(), 'vector_result')
        if not os.path.exists(vector_path):
            os.mkdir(vector_path)
        for batchi, (imgs, trans, rots, intrins, post_trans, post_rots, car_trans, yaw_pitch_roll, segmentation_gt, instance_gt, direction_gt, sample_token) in enumerate(val_loader):
            if torch.cuda.is_available():
                segmentation, embedding, direction = model(imgs.cuda(), trans.cuda(), rots.cuda(), intrins.cuda(),
                                                       post_trans.cuda(), post_rots.cuda(),
                                                       car_trans.cuda(), yaw_pitch_roll.cuda(), sample_token[0] )
            else:
                segmentation, embedding, direction = model(imgs, trans, rots, intrins,
                                                       post_trans, post_rots, lidar_data,
                                                       lidar_mask, car_trans, yaw_pitch_roll, sample_token[0])
            sample_token_str = sample_token[0]
            print(segmentation.shape)
            
            # si is the batch size, == 0 in debug.
            for si in range(segmentation.shape[0]):
                coords, _, _ = vectorize(segmentation[si], embedding[si], direction[si], angle_class)

                for coord in coords:
                    #draw lines based on its x , y and lane width
                    plt.plot(coord[:, 0], coord[:, 1], linewidth=5)

                plt.xlim((0, segmentation.shape[3]))
                plt.ylim((0, segmentation.shape[2]))
                plt.imshow(car_img, extent=[segmentation.shape[3]//2-15, segmentation.shape[3]//2+15, segmentation.shape[2]//2-12, segmentation.shape[2]//2+12])

                # img_name = f'eval_vector_{batchi:06}_{sample_token_str}_{si:03}.jpg'
                # image_path = os.path.join(vector_path, img_name)
                # print('saving', image_path)
                # plt.savefig(image_path)
                # plt.close()
                img_name = f'plt_images/{sample_token_str}/eval_vector{batchi:06}_{sample_token_str}_{si:03}.jpg'                
                print('saving', img_name)
                plt.savefig(img_name)
                plt.close()
                
                # vis_resutls(args, img_name, batchi, sample_token_str)


def main(args):
    data_conf = {
        'num_channels': NUM_CLASSES + 1,
        'image_size': args.image_size,
        'xbound': args.xbound,
        'ybound': args.ybound,
        'zbound': args.zbound,
        'dbound': args.dbound,
        'tf_size': args.tf_size,
        'thickness': args.thickness,
        'angle_class': args.angle_class,
    }

    train_loader_tf, val_loader_tf = semantic_dataset_tf(args.version, args.dataroot, data_conf, args.bsz, args.nworkers, args.tf_size)
    model = get_model(args.model, data_conf, args.instance_seg, args.embedding_dim, args.direction_pred, args.angle_class)
    model.load_state_dict(torch.load(args.modelf), strict=False)
    if torch.cuda.is_available():
        model.cuda()
        
    vis_segmentation(model, val_loader_tf)
    # vis_vector(model, val_loader, args.angle_class)
    # vis_segmentation(model, val_loader)
    

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    # logging config
    parser.add_argument("--logdir", type=str, default='./runs')

    # nuScenes config
    parser.add_argument('--dataroot', type=str, default='dataset/nuScenes/')
    parser.add_argument('--version', type=str, default='v1.0-mini', choices=['v1.0-trainval', 'v1.0-mini', 'mb_test'])

    # model config
    parser.add_argument("--model", type=str, default='HDMapNet_cam')

    # training config
    parser.add_argument("--nepochs", type=int, default=30)
    parser.add_argument("--max_grad_norm", type=float, default=5.0)
    parser.add_argument("--pos_weight", type=float, default=2.13)
    parser.add_argument("--bsz", type=int, default=1)
    parser.add_argument("--nworkers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-7)

    # finetune config
    parser.add_argument('--finetune', action='store_true')
    parser.add_argument('--modelf', type=str, default=None)

    # data config
    parser.add_argument("--thickness", type=int, default=5)
    parser.add_argument("--image_size", nargs=2, type=int, default=[128, 352])
    parser.add_argument("--xbound", nargs=3, type=float, default=[-30.0, 30.0, 0.15])
    parser.add_argument("--ybound", nargs=3, type=float, default=[-15.0, 15.0, 0.15])
    parser.add_argument("--zbound", nargs=3, type=float, default=[-10.0, 10.0, 20.0])
    parser.add_argument("--dbound", nargs=3, type=float, default=[4.0, 45.0, 1.0])
    parser.add_argument("--tf_size", type=int, default=1)


    # embedding config
    parser.add_argument('--instance_seg', action='store_true')
    parser.add_argument("--embedding_dim", type=int, default=16)
    parser.add_argument("--delta_v", type=float, default=0.5)
    parser.add_argument("--delta_d", type=float, default=3.0)

    # direction config
    parser.add_argument('--direction_pred', action='store_true')
    parser.add_argument('--angle_class', type=int, default=36)

    # loss config
    parser.add_argument("--scale_seg", type=float, default=1.0)
    parser.add_argument("--scale_var", type=float, default=1.0)
    parser.add_argument("--scale_dist", type=float, default=1.0)
    parser.add_argument("--scale_direction", type=float, default=0.2)

    args = parser.parse_args()
    main(args)
