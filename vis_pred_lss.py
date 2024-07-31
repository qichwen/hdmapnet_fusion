import argparse
import shutil
import numpy as np
from PIL import Image
import os

import matplotlib.pyplot as plt

import tqdm
import torch
from time import time
import gc

from data.dataset import semantic_dataset
from data.const import NUM_CLASSES
from model import get_model
from postprocess.vectorize import vectorize
from datetime import datetime  
import logging
  
def onehot_encoding(logits, dim=1):
    max_idx = torch.argmax(logits, dim, keepdim=True)
    one_hot = logits.new_full(logits.shape, 0)
    one_hot.scatter_(dim, max_idx, 1)
    return one_hot

# def vis_segmentation(model, val_loader):
#     model.eval()
#     with torch.no_grad():
#         for batchi, (imgs, trans, rots, intrins, post_trans, post_rots, lidar_data, lidar_mask, car_trans, yaw_pitch_roll, semantic_gt, instance_gt, direction_gt) in enumerate(val_loader):
#             if torch.cuda.is_available():
#                 semantic, embedding, direction = model(imgs.cuda(), trans.cuda(), rots.cuda(), intrins.cuda(),
#                                                     post_trans.cuda(), post_rots.cuda(), lidar_data.cuda(),
#                                                     lidar_mask.cuda(), car_trans.cuda(), yaw_pitch_roll.cuda())
#                 semantic = semantic.softmax(1).cpu().numpy()
#             else:
#                 semantic, embedding, direction = model(imgs, trans, rots, intrins,
#                                                     post_trans, post_rots, lidar_data,
#                                                     lidar_mask, car_trans, yaw_pitch_roll)
#                 semantic = semantic.softmax(1).numpy()
#             semantic[semantic_gt < 0.1] = np.nan

#             for si in range(semantic.shape[0]):
#                 plt.figure(figsize=(4, 2))
#                 plt.imshow(semantic[si][1], vmin=0, cmap='Blues', vmax=1, alpha=0.8)
#                 plt.imshow(semantic[si][2], vmin=0, cmap='Reds', vmax=1, alpha=0.8)
#                 plt.imshow(semantic[si][3], vmin=0, cmap='Greens', vmax=1, alpha=0.8)

#                 # fig.axes.get_xaxis().set_visible(False)
#                 # fig.axes.get_yaxis().set_visible(False)
#                 plt.xlim(0, 400)
#                 plt.ylim(0, 200)
#                 plt.axis('off')

#                 imname = f'eval{batchi:06}_{si:03}.jpg'
#                 print('saving', imname)
#                 plt.savefig(imname)
#                 plt.close()

def vis_segmentation_lss(model, val_loader, pltimage_result_dir, logger):
    model.eval()
    with torch.no_grad():
        segment_path = os.path.join(os.getcwd(), 'segment_result')    
        if not os.path.exists(segment_path):
            os.mkdir(segment_path)
        pred_cur = time()
        # for batchi, (imgs, trans, rots, intrins, post_trans, post_rots, lidar_data, lidar_mask, car_trans, yaw_pitch_roll, semantic_gt, instance_gt, direction_gt, sample_token) in enumerate(val_loader):        
        for batchi, (imgs, trans, rots, intrins, post_trans, post_rots, sample_token) in enumerate(val_loader):
            pred_sts = time()
            if torch.cuda.is_available():
                semantic, embedding, direction = model(imgs.cuda(), trans.cuda(), rots.cuda(), intrins.cuda(),
                                                    post_trans.cuda(), post_rots.cuda(), sample_token)
                # forward(self, x, trans, rots, intrins, post_trans, post_rots, sample_token)
                # semantic, embedding, direction = model(imgs.cuda(), trans.cuda(), rots.cuda(), intrins.cuda(),
                #                                     post_trans.cuda(), post_rots.cuda(), lidar_data.cuda(),
                #                                     lidar_mask.cuda(), car_trans.cuda(), yaw_pitch_roll.cuda())
                semantic = semantic.softmax(1).cpu().numpy()
                # semantic = semantic.sigmoid().cpu()
            else:
                semantic, embedding, direction = model(imgs, trans, rots, intrins,
                                                    post_trans, post_rots, lidar_data,
                                                    lidar_mask, car_trans, yaw_pitch_roll,sample_token[0])
                semantic = semantic.softmax(1).numpy()
            
            pred_ets = time()
            
            # out = out.sigmoid().cpu()
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
            sample_token_str = sample_token[0]
            pltimage_dir = os.path.join(pltimage_result_dir, sample_token_str)
            
            #draw
            for si in range(semantic.shape[0]):
                #print(len(semantic[0][1]))
                #print(semantic[0][1])
                if not os.path.exists(pltimage_dir):
                    os.mkdir(pltimage_dir)
                plt.figure(figsize=(4, 2), dpi=400)
                
                plt.imshow(semantic[si][0], alpha=0.8)
                plt.xlim(0, 400)
                plt.ylim(0, 200)
                #plt.axis('off')
                imname = f'segment_LSS_all_{sample_token_str}.jpg'
                image_path = os.path.join(pltimage_dir, imname)                
                print('saving', image_path)                
                plt.savefig(image_path)
                # plt.close()
                
                plt.imshow(semantic[si][1], vmin=0, cmap='Blues', vmax=1, alpha=0.8)
                plt.xlim(0, 400)
                plt.ylim(0, 200)
                #plt.axis('off')
                imname = f'segment_LSS_addedDivider_{sample_token_str}.jpg'
                image_path = os.path.join(pltimage_dir, imname)                
                print('saving', image_path)
                
                plt.savefig(image_path)
                # plt.close()
                
                plt.imshow(semantic[si][2], vmin=0, cmap='Reds', vmax=1, alpha=0.8)
                plt.xlim(0, 400)
                plt.ylim(0, 200)
                plt.axis('off')
                imname = f'segment_LSS_addPedestrain_{sample_token_str}.jpg'
                image_path = os.path.join(pltimage_dir, imname)                
                print('saving', image_path)
                #plt.savefig(image_path)
                # plt.close()
                
                plt.imshow(semantic[si][3], vmin=0, cmap='Greens', vmax=1, alpha=0.8)
                plt.xlim(0, 400)
                plt.ylim(0, 200)
                #plt.axis('off')
                imname = f'segment_LSS_addedContour_{sample_token_str}.jpg'
                image_path = os.path.join(pltimage_dir, imname)                
                print('saving', image_path)
                plt.savefig(image_path)
                plt.close()

                # #seg gt overall geometry
                # plt.imshow(semantic_gt[si][0],alpha=0.8)
                # plt.xlim(0, 400)
                # plt.ylim(0, 200)
                # plt.axis('off')
                # imname = f'segmentGT_LSS_[0][0]_{sample_token_str}.jpg'
                # image_path = os.path.join(pltimage_dir, imname)                
                # print('saving', image_path)
                # plt.savefig(image_path)
                # plt.close()
                
                # # road / lane divider
                # plt.imshow(semantic_gt[si][1], alpha=0.8)
                # plt.xlim(0, 400)
                # plt.ylim(0, 200)
                # plt.axis('off')
                # imname = f'segmentGT_LSS_[0][1]_{sample_token_str}.jpg'
                # image_path = os.path.join(pltimage_dir, imname)                
                # print('saving', image_path)
                # plt.savefig(image_path)
                # plt.close()
                
                # # pedestrain crossing
                # plt.imshow(semantic_gt[si][2], alpha=0.8)
                # plt.xlim(0, 400)
                # plt.ylim(0, 200)
                # plt.axis('off')
                # imname = f'segmentGT_LSS_[0][2]_{sample_token_str}.jpg'
                # image_path = os.path.join(pltimage_dir, imname)                
                # print('saving', image_path)
                # plt.savefig(image_path)
                # plt.close()
                
                # # contour
                # plt.imshow(semantic_gt[si][3], alpha=0.8)
                # plt.xlim(0, 400)
                # plt.ylim(0, 200)
                # plt.axis('off')
                # imname = f'segmentGT_LSS_[0][3]_{sample_token_str}.jpg'
                # image_path = os.path.join(pltimage_dir, imname)                
                # print('saving', image_path)
                # plt.savefig(image_path)
                # plt.close()
                
                # fig.axes.get_xaxis().set_visible(False)
                # fig.axes.get_yaxis().set_visible(False)
                plt.xlim(0, 400)
                plt.ylim(0, 200)
                plt.axis('off')
                #TODO: print seg gt
                imname = f'eval_segment_LSS_{batchi:06}_{sample_token_str}_{si:03}.jpg'
                image_path = os.path.join(segment_path, imname)
                print('saving', image_path)
                plt.savefig(image_path)
                
                pltimage_path = f'{pltimage_dir}/segmentation_LSS_{batchi:06}_{sample_token_str}_{si:03}.jpg'
                plt.savefig(pltimage_path)                
                # fig.clf()
                plt.close()                
                gc.collect()
            
            plt.close()            
            
            del semantic
            gc.collect()
            draw_e = time()
            pred_cur = draw_e
            logger.info(f"====== Batchi '{batchi:>3d}' segment timing as below ======    "
                        f"Batchi: [{batchi:>3d}]    "
                        f"Dataload Time: {pred_cur - pred_sts:>7.4f}    "
                        f"Pred Time: {pred_ets-pred_sts:>7.4f}    "
                        f"Draw Time: {draw_e-pred_ets:>7.4f}    "
                        f"    "
                        )
            
#mb_test
def vis_segmentation(model, val_loader, draw, pltimage_result_dir, logging):
    model.eval()
    segment_path = os.path.join(os.getcwd(), pltimage_result_dir, 'segment_result')
    if not os.path.exists(segment_path):
        os.mkdir(segment_path)
    
    with torch.no_grad():
        for batchi, (imgs, trans, rots, intrins, post_trans, post_rots, sample_token) in enumerate(val_loader):
            if torch.cuda.is_available():
                semantic, embedding, direction = model(imgs.cuda(), trans.cuda(), rots.cuda(), intrins.cuda(),
                                                    post_trans.cuda(), post_rots.cuda(),sample_token[0], pltimage_result_dir, draw)
                semantic = semantic.softmax(1).cpu().numpy()
            else:
                #mb test add samp_token
                semantic, embedding, direction = model(imgs, trans, rots, intrins,
                                                    post_trans, post_rots, sample_token[0], draw)
                semantic = semantic.softmax(1).numpy()
            # mb_test, comment for continue
            # semantic[semantic_gt < 0.1] = np.nan
            sample_token_str = sample_token[0]

            for si in range(semantic.shape[0]):
                # * (400, 200) = 1600,800
                plt.figure(figsize=(4, 2))
                plt.imshow(semantic[si][1], vmin=0, cmap='Blues', vmax=1, alpha=0.8)
                plt.imshow(semantic[si][2], vmin=0, cmap='Reds', vmax=1, alpha=0.8)
                plt.imshow(semantic[si][3], vmin=0, cmap='Greens', vmax=1, alpha=0.8)

                # fig.axes.get_xaxis().set_visible(False)
                # fig.axes.get_yaxis().set_visible(False)
                plt.xlim(0, 400)
                plt.ylim(0, 200)
                #   plt.axis('off')

                imname = f'eval_segment_{batchi:06}_{sample_token_str}_{si:03}.jpg'
                image_path = os.path.join(segment_path, imname)
                print('saving', image_path)
                
                
                
                smp_dir = os.path.join(pltimage_result_dir, sample_token_str)
                #print(os.path.exists(pltimage_dir))
                if not os.path.exists(smp_dir):
                    os.mkdir(smp_dir)
                
                pltimage_path = f'{smp_dir}/segmentation_{batchi:06}_{sample_token_str}_{si:03}.jpg'
                plt.savefig(pltimage_path)
                plt.savefig(image_path)
                plt.close()


def vis_vector(model, val_loader, angle_class, draw, pltimage_result_dir, logger):
    model.eval() #evaluation mode
    car_img = Image.open('icon/car.png')
    
    with torch.no_grad():
        # for batchi, (imgs, trans, rots, intrins, post_trans, post_rots, lidar_data, lidar_mask, car_trans, yaw_pitch_roll, segmentation_gt, instance_gt, direction_gt) in enumerate(val_loader):

        for batchi, (imgs, trans, rots, intrins, post_trans, post_rots, sample_token) in enumerate(val_loader):
            
            if torch.cuda.is_available():
                segmentation, embedding, direction = model(imgs.cuda(), trans.cuda(), rots.cuda(), intrins.cuda(),
                                                    post_trans.cuda(), post_rots.cuda(), sample_token)
            else:
                # segmentation, embedding, direction = model(imgs, trans, rots, intrins,
                #                                        post_trans, post_rots, lidar_data,
                #                                        lidar_mask, car_trans, yaw_pitch_roll)
                # re-define the model forward func
                # segmentation, embedding, direction = model(imgs, trans, rots, intrins,
                #                                        post_trans, post_rots)
                #mb test add samp_token
                segmentation, embedding, direction = model(imgs, trans, rots, intrins,
                                                    post_trans, post_rots, sample_token[0], draw)
                
            sample_token_str = sample_token[0]

            print(segmentation.shape)
            # 1, 4, 200, 400
            # si - for eacy batchi
            
            vect_cur = time()
            for si in range(segmentation.shape[0]):                
                coords, _, _ = vectorize(segmentation[si], embedding[si], direction[si], angle_class)
                #for each lane shaped no matter its class, 1 color would be assigned for each sample
                vectorized_ets = time()
                for coord in coords:
                    plt.plot(coord[:, 0], coord[:, 1], linewidth=5)
                                
                plt.xlim((0, segmentation.shape[3]))
                plt.ylim((0, segmentation.shape[2]))
                
                plt.imshow(car_img, extent=[segmentation.shape[3]//2-15, segmentation.shape[3]//2+15, segmentation.shape[2]//2-12, segmentation.shape[2]//2+12])                
                img_name = f'{pltimage_result_dir}/{sample_token_str}/eval_vector{batchi:06}_{sample_token_str}_{si:03}.jpg'                
                print('saving', img_name)
                plt.savefig(img_name)
                plt.close()
                vect_draw_ets = time()
                vis_resutls(args, img_name, batchi, sample_token_str, pltimage_result_dir)
                test_cluster_ets = time()                
            
                logger.info(f"====== vectorized timing for each batch ======    "
                    f"Batchi: [{batchi:>3d}]    "
                    f"sample index of above batch: [{si:>3d}]    "
                    #f"Dataload Time: {pred_sts - pred_cur:>7.4f}    "
                    f"AllClass Vectorized computing time: {vectorized_ets-vect_cur:>7.4f}    "
                    f"vect draw Time: {vect_draw_ets-vectorized_ets:>7.4f}    "
                    f"cluster Time: {test_cluster_ets-vect_draw_ets:>7.4f}    "
                    f"    "
                    )
                vect_cur = test_cluster_ets
                

def vis_resutls(args, img_name, batchi, sample_token_str, pltimage_result_dir):
    batchi = 0
    # cluster imge path
    img_path = f'{pltimage_result_dir}/{sample_token_str}/'
    # cluster_path = os.path.join(img_path, 'image_cluster')
    if not os.path.exists(img_path):
        os.mkdir(img_path)
    smaple_token = sample_token_str
    # samples = read_json_file(args.image_json)
    # for smaple_token in samples:
    #TODO: to add try exception logic; - done
    new_cam_image = Image.new('RGB', (10000, 5000))

    BEV_topdown = [f for f in os.listdir(img_path) if (f.startswith('topdown_afterIPM_Usamp_')) and f.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))]
    if len(BEV_topdown) != 0:        
        BEV_topdown = os.path.join(img_path, BEV_topdown[0])
        bev_img_cam_ori = Image.open(BEV_topdown)
        bev_img_cam = bev_img_cam_ori.resize((2200, 1200))
        new_cam_image.paste(bev_img_cam, (0, 0))
    
    segment_img = [f for f in os.listdir(img_path) if (f.startswith('segment_LSS_addedContour')) and f.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))]
    if len(segment_img) != 0:     
        segment_img = os.path.join(img_path, segment_img[0])
        seg_img_cam_ori = Image.open(segment_img)
        seg_img_cam = seg_img_cam_ori.resize((2200, 1200))
        new_cam_image.paste(seg_img_cam, (4400, 0))
        
    vector_img = [f for f in os.listdir(img_path) if (f.startswith('eval_')) and f.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))]
    if len(vector_img) != 0:
        vector_img = os.path.join(img_path, vector_img[0])
        vec_img_cam_ori = Image.open(vector_img)
        vec_img_cam = vec_img_cam_ori.resize((1600, 1200))
        new_cam_image.paste(vec_img_cam, (6600, 0))

    overall_img = [f for f in os.listdir(img_path) if (f.startswith('6viewsfused')) and f.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))]
    if len(overall_img) != 0: 
        overall_img = os.path.join(img_path, overall_img[0])
        overall_img_ori = Image.open(overall_img)
        overall_img_cam = overall_img_ori.resize((6000, 3600))
        new_cam_image.paste(overall_img_cam, (0, 1200+1200*2))   
   
    seg_bev_img_ori = [f for f in os.listdir(img_path) if (f.startswith('fm_overall_')) and f.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))]
    if len(seg_bev_img_ori) != 0: 
        seg_bev_img = os.path.join(img_path, seg_bev_img_ori[0])
        seg_bev_ori = Image.open(seg_bev_img)
        seg_bev = seg_bev_ori.resize((2200, 1200))
        new_cam_image.paste(seg_bev, (2200, 0))
    #
    ###pastes the overall_img image onto the new_cam_image at the specified position (0, 1800). This means that 
    # the top-left corner of overall_img will be placed at the x-coordinate 0 and the y-coordinate 1800 on the new_cam_image   
    
    # plt.imshow(bev_img_cam)
    # plt.axis('off')
    # plt.savefig(f'BEV_topdown_.jpg')
    
    # plt.imshow(seg_img_cam)
    # plt.axis('off')
    # plt.savefig(f'seg_.jpg')
    
    # plt.imshow(vec_img_cam)
    # plt.axis('off')
    # plt.savefig(f'vec_.jpg')
             
    # if not os.path.exists(segment_sample_path):
    #     os.mkdir(segment_sample_path)
    # shutil.copy(segment_img, segment_sample_path)
        
    raw_imgs = sorted([f for f in os.listdir(img_path) if (((('fm_imgafter_norm') in f) or ('pvimg_before_norm' in f)) and f.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')))])  
    for imgn, image_file in enumerate(raw_imgs):
        rows = imgn//6
        rols = imgn%6  
        image_path = os.path.join(img_path, image_file)                  
        pv = Image.open(image_path).resize((1600,1200)) 
        new_cam_image.paste(pv, (rols * 1600, 1200+1200*rows))      
    
    img_dir = os.path.join(pltimage_result_dir,'cluster')
       
    if not os.path.exists(img_dir):
        os.mkdir(img_dir)
    
    img_name = f'cluster_{batchi:06}_{smaple_token}.jpg'    
    new_image_path = os.path.join(img_dir,img_name)       
    new_cam_image.save(new_image_path)
    shutil.copy(new_image_path, img_path)
    # batchi = batchi + 1    
    print(f"exported to {new_image_path}")

def main(args):
    data_conf = {
        'num_channels': NUM_CLASSES + 1,
        'image_size': args.image_size,
        'xbound': args.xbound,
        'ybound': args.ybound,
        'zbound': args.zbound,
        'dbound': args.dbound,
        'thickness': args.thickness,
        'angle_class': args.angle_class,
    }
    
    now = datetime.now() 
    
    
    if not os.path.exists(args.predlogdir):
        os.mkdir(args.predlogdir)
    logging.basicConfig(filename=os.path.join(args.predlogdir, "pred.log"),
                        filemode='w',
                        format='%(asctime)s: %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        level=logging.INFO)
    logging.getLogger('shapely.geos').setLevel(logging.CRITICAL)

    logger = logging.getLogger()
    logger.addHandler(logging.StreamHandler(sys.stdout))

    
    if not os.path.exists('plt_images'):
        os.mkdir('plt_images')
        
    if args.lss:        
        if not os.path.exists(args.lss):
            os.mkdir(args.lss)
            
    dhms = f'{now.year}_{now.month}_{now.day}_{now.hour}_{now.minute}_{now.second}'
    
    if args.lss:
        pltimage_result_dir = os.path.join(args.lss, dhms)
    else:
        pltimage_result_dir = os.path.join('plt_images', dhms)
        
    if not os.path.exists(pltimage_result_dir):
        os.mkdir(pltimage_result_dir)
    
    train_loader, val_loader = semantic_dataset(args.version, args.dataroot, data_conf, args.bsz, args.nworkers, pltimage_result_dir)
    model = get_model(pltimage_result_dir, args.model, data_conf, args.instance_seg, args.embedding_dim, args.direction_pred, args.angle_class)
    
    model.load_state_dict(torch.load(args.modelf), strict=False)
    if torch.cuda.is_available():
        model.cuda()
     
    # pltimage_result_dir = os.path.join('plt_images', dhms)
    # if not os.path.exists(pltimage_result_dir):
    #     os.makedir(pltimage_result_dir)
    
    vis_segmentation_lss(model, val_loader, pltimage_result_dir, logger)
    
    vis_vector(model, val_loader, args.angle_class, args.draw, pltimage_result_dir, logger)
    # vis
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    import sys
    print("ptyhnon env:")
    print(sys.version)
    # logging config
    parser.add_argument('--seg_path', type=str, default='segment_result/')
    parser.add_argument('--vec_path', type=str, default='')
    parser.add_argument('--image_json', type=str, default='/samples.json')
    parser.add_argument('--output', type=str, default='/')
    
    parser.add_argument("--logdir", type=str, default='./runs')
    parser.add_argument("--predlogdir", type=str, default='plt_images/lss/')
    parser.add_argument("--draw", type=str, default=None)

    # nuScenes config
    parser.add_argument('--dataroot', type=str, default='dataset/nuScenes/')
    parser.add_argument('--version', type=str, default='v1.0-mini', choices=['v1.0-trainval', 'v1.0-mini', 'mb_test'])

    # model config
    parser.add_argument("--model", type=str, default='HDMapNet_cam')
    parser.add_argument('--lss', type=str, default='plt_images/lss/')

    # training config
    parser.add_argument("--nepochs", type=int, default=30)
    parser.add_argument("--max_grad_norm", type=float, default=5.0)
    parser.add_argument("--pos_weight", type=float, default=2.13)
    # set bsz = 1, nworkers = 0
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
