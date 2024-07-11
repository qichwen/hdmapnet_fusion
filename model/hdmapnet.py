import torch
from torch import nn

from .homography import bilinear_sampler, IPM
from .utils import plane_grid_2d, get_rot_2d, cam_to_pixel
from .pointpillar import PointPillarEncoder
from .base import CamEncode, BevEncode
from data.utils import gen_dx_bx
import matplotlib.pyplot as plt
import gc
import os
import psutil

# from image_processer import visualizer_plt




class ViewTransformation(nn.Module):
    def __init__(self, fv_size, bv_size, n_views=6):
        super(ViewTransformation, self).__init__()
        self.n_views = n_views
        self.hw_mat = []
        self.bv_size = bv_size
        fv_dim = fv_size[0] * fv_size[1]
        bv_dim = bv_size[0] * bv_size[1]
        for i in range(self.n_views):
            fc_transform = nn.Sequential(
                nn.Linear(fv_dim, bv_dim),
                nn.ReLU(),
                nn.Linear(bv_dim, bv_dim),
                nn.ReLU()
            )
            self.hw_mat.append(fc_transform)
        self.hw_mat = nn.ModuleList(self.hw_mat)

    def forward(self, feat):
        B, N, C, H, W = feat.shape
        feat = feat.view(B, N, C, H*W)
        outputs = []
        for i in range(N):
            output = self.hw_mat[i](feat[:, i]).view(B, C, self.bv_size[0], self.bv_size[1])
            outputs.append(output)
        outputs = torch.stack(outputs, 1)
        return outputs


class HDMapNet(nn.Module):
    def __init__(self, data_conf, instance_seg=True, embedded_dim=16, direction_pred=True, direction_dim=36, lidar=False):
        super(HDMapNet, self).__init__()
        self.camC = 64
        self.downsample = 16

        dx, bx, nx = gen_dx_bx(data_conf['xbound'], data_conf['ybound'], data_conf['zbound'])
        # dx: tensor([ 0.1500,  0.1500, 20.0000])
        # bx: tensor([-29.9250, -14.9250,   0.0000])
        # nx: tensor([400, 200,   1])
        final_H, final_W = nx[1].item(), nx[0].item()

        self.camencode = CamEncode(self.camC)
        fv_size = (data_conf['image_size'][0]//self.downsample, data_conf['image_size'][1]//self.downsample)
        bv_size = (final_H//5, final_W//5)
        self.view_fusion = ViewTransformation(fv_size=fv_size, bv_size=bv_size)

        res_x = bv_size[1] * 3 // 4
        ipm_xbound = [-res_x, res_x, 4*res_x/final_W]
        #[-60, 60, 0.6]
        ipm_ybound = [-res_x/2, res_x/2, 2*res_x/final_H]
        #[-30.0, 30.0, 0.6]
        
        self.ipm = IPM(ipm_xbound, ipm_ybound, N=6, C=self.camC, visual=True, extrinsic=True)
        self.up_sampler = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        # self.up_sampler = nn.Upsample(scale_factor=5, mode='bilinear', align_corners=True)

        self.lidar = lidar
        if lidar:
            self.pp = PointPillarEncoder(128, data_conf['xbound'], data_conf['ybound'], data_conf['zbound'])
            self.bevencode = BevEncode(inC=self.camC+128, outC=data_conf['num_channels'], instance_seg=instance_seg, embedded_dim=embedded_dim, direction_pred=direction_pred, direction_dim=direction_dim+1)
        else:
            self.bevencode = BevEncode(inC=self.camC, outC=data_conf['num_channels'], instance_seg=instance_seg, embedded_dim=embedded_dim, direction_pred=direction_pred, direction_dim=direction_dim+1)

    def get_Ks_RTs_and_post_RTs(self, intrins, rots, trans, post_rots, post_trans):
        B, N, _, _ = intrins.shape
        # reshape to B,N,4,4
        Ks = torch.eye(4, device=intrins.device).view(1, 1, 4, 4).repeat(B, N, 1, 1)

        Rs = torch.eye(4, device=rots.device).view(1, 1, 4, 4).repeat(B, N, 1, 1)
        Rs[:, :, :3, :3] = rots.transpose(-1, -2).contiguous()
        Ts = torch.eye(4, device=trans.device).view(1, 1, 4, 4).repeat(B, N, 1, 1)
        Ts[:, :, :3, 3] = -trans
        RTs = Rs @ Ts

        post_RTs = None

        return Ks, RTs, post_RTs

    def get_cam_feats(self, x):
        B, N, C, imH, imW = x.shape
        x = x.view(B*N, C, imH, imW)
        
        x = self.camencode(x)
        x = x.view(B, N, self.camC, imH//self.downsample, imW//self.downsample)
        return x

    def forward(self, img, trans, rots, intrins, post_trans, post_rots, lidar_data, lidar_mask, car_trans, yaw_pitch_roll, samp_token):
        #x = self.get_cam_feats(img)        
        # x = self.view_fusion(x)
        # Ks, RTs, post_RTs = self.get_Ks_RTs_and_post_RTs(intrins, rots, trans, post_rots, post_trans)
        # topdown = self.ipm(x, Ks, RTs, car_trans, yaw_pitch_roll, post_RTs)
        # topdown = self.up_sampler(topdown)
        # if self.lidar:
        #     lidar_feature = self.pp(lidar_data, lidar_mask)
        #     topdown = torch.cat([topdown, lidar_feature], dim=1)
        # return self.bevencode(topdown)
        
        process = psutil.Process(os.getpid())
        print('BeforeFigure:', process.memory_info().rss)  # in bytes
        
        #TODO: controll by param
        draw = False
        
        imgs_path = os.path.join('plt_images', samp_token)
        if not os.path.exists(imgs_path):
            os.makedirs(imgs_path)
        
        if draw:
            fig = plt.figure(figsize=(60, 40))
            file_path = f"plt_images/{samp_token}"             
            raw_imgs = sorted([f for f in os.listdir(file_path) if (((('fm_imgafter_norm') in f) or ('pvimg_before_norm' in f)) and f.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')))])  
            
            rows = 8
            cols = 6
            imgcounter = 0
            for imgn, image_file in enumerate(raw_imgs):  
                full_path = os.path.join(file_path, image_file)                  
                raw_img = plt.imread(full_path) 
                loc = imgn + 1  
                a = fig.add_subplot(rows, cols, loc)  
                a.set_title(f'{image_file}')
                # a.axis('off')
                # plt.axis('off')
                plt.imshow(raw_img)
                # plt.show()
                # plt.axis('off')
                imgcounter += 1
                
                # 调整子图之间的间距  
                # plt.subplots_adjust(wspace=0.5, hspace=0.5)
                print('Before close:', process.memory_info().rss)  # in bytes
                plt.savefig(f'plt_images/{samp_token}/pvs_norm.png')
            plt.close(fig)
            # a.remove()
            # del a
            plt.clf()
            plt.cla()
            del fig
            gc.collect()
        print('pvs_norm:', process.memory_info().rss)  # in bytes

        #CAMS = ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT']
        x = self.get_cam_feats(img)
        #(1,6,64,8,22)        
        #store imgs of all chns for each of the 6 views after CameraEncoder 
        # for view in range(x.shape[1]):
        #     for chn in range(x.shape[2]):
        #         visualizer_plt(x[0][view][chn], step=f"afterCamencode_v{view}_chn{chn}")

        x = self.view_fusion(x)
        # (1, 6, 64, 40, 80)
        #store imgs of afterviewtrans all chns for each of the 6 views
        # for view in range(x.shape[1]):
        #     for chn in range(x.shape[2]):
        #         visualizer_plt(x[0][view][chn], step=f"afterviewtransform_v{view}_chn{chn}")

        # worked in the size 20 * 20 inchs, 1000 * 1000 pixels
        if draw:
            for view in range(x.shape[1]):
                vt_fm = torch.sum(x[0][view].squeeze(0),0)
                print(vt_fm.shape) 
                imgcounter += 1
                # Layout the imgs by view
                # loc = int(str(53)+str(view+1))  
                loc = imgcounter      
                a = fig.add_subplot(rows, cols, loc)                
                # a.axis('off')
                a.set_title(f'6viewsfused_{view}')
                plt.imshow(vt_fm.detach().cpu().numpy())
                
            plt.savefig(f'plt_images/{samp_token}/6viewsfused.png')  
            plt.close(fig)
            print('6viewsfused:', process.memory_info().rss)  # in bytes

        # plt.savefig(f'{imgs_path}/vtfm_6views_into1.png') 

        # fig = plt.figure(figsize=(20, 20))
        # for view in range(x.shape[1]):
        #     vt_fm = torch.sum(x[0][view].squeeze(0),0)
        #     print(vt_fm.shape) 
        #     imgcounter += 1
        #     # Layout the imgs by view
        #     loc = int(str(53)+str(view+1))  
        #     # loc = imgcounter      
        #     a = fig.add_subplot(loc)                
        #     a.axis('off')
        #     a.set_title(f'6viewsfused_{view}')
        #     plt.imshow(vt_fm.detach().cpu().numpy())
                
        # for view in range(x.shape[1]):
        #     vt_fm = torch.sum(x[0][view].squeeze(0),0)
        #     print(vt_fm.shape) 
        #     # Layout the imgs by view
        #     loc = int(str(53)+str(view+1))       
        #     a = fig.add_subplot(loc)
        #     # not worked by size assigned 
        #     # a = fig.add_subplot(50 ,100 , int(view+1))
        #     # a.axis('off')
        #     a.set_title(f'vtfm_v{view}')
        #     plt.imshow(vt_fm)
        #     imgcounter += 1
        
        Ks, RTs, post_RTs = self.get_Ks_RTs_and_post_RTs(intrins, rots, trans, post_rots, post_trans)
        topdown = self.ipm(x, Ks, RTs,samp_token, car_trans, yaw_pitch_roll, post_RTs)
        # torch.Size([1, 64, 100, 200])
        
        if draw:
            fm = torch.sum(topdown.squeeze(0),0)
            # print(fm.shape)
            imgcounter += 1
            loc = imgcounter      
            a = fig.add_subplot(rows, cols, loc)  
            plt.imshow(fm.detach().cpu().numpy())
            a.set_title('topdown_afterIPM')
            plt.savefig(f'plt_images/{samp_token}/topdown_afterIPM.png', bbox_inches='tight')
            plt.close(a)
        topdown = self.up_sampler(topdown)
        # torch.Size([1, 64, 200, 400])
        
        if draw:        
            #store imgs of all 64 chns
            # for chn in range(topdown[0].shape[0]):
            #     visualizer_plt(topdown[0][chn], step=f"afteripm_chn_{chn}")
            fm = torch.sum(topdown.squeeze(0),0)
            plt.imshow(fm.detach().cpu().numpy())
            plt.savefig(f'plt_images/{samp_token}/topdown_afterIPM_Usamp_{samp_token}.png', bbox_inches='tight')  
            print(fm.shape)
            # 5 rows, 3 cols, index = imgcounter
            imgcounter += 1
            loc = imgcounter
            a.set_title('topdown_afterIPM')
            a = fig.add_subplot(rows, cols, loc) 
            # a.axis('off')           
            # a.set_title('BEV_topdown')
            plt.savefig(f'plt_images/{samp_token}/topdown_afterIPM_Usamp_{samp_token}.png', bbox_inches='tight')               
            plt.close(a)
        # fm = torch.sum(topdown.squeeze(0),0)
        # print(fm.shape)
        # # 5 rows, 3 cols, index = imgcounter
        # a = fig.add_subplot(5,3,imgcounter)
        # plt.imshow(fm.detach().cpu().numpy())
        # a.set_title('Aft_VTup_in_1')
        # plt.savefig(f'{imgs_path}/Aft_VTup_in_1.png', bbox_inches='tight')
        # imgcounter += 1
        # if self.lidar:
        #     lidar_feature = self.pp(lidar_data, lidar_mask)
        #     topdown = torch.cat([topdown, lidar_feature], dim=1)
        # print(topdown)
        if draw:
            seg, emb, dir = self.bevencode(topdown)
            #seg : (1,4, 200, 400)
            segfm = torch.sum(seg.squeeze(0),0)
            print(segfm.shape)
            # 5 rows, 3 cols, index = imgcounter
            loc = imgcounter + 1      
            a = fig.add_subplot(rows, cols, loc)
            # a.axis('off')
            # plt.axis('off')
            plt.imshow(segfm.detach().cpu().numpy())
            a.set_title('segfm_BEV_topdown')
            imgcounter += 1          
                        
            plt.savefig(f'plt_images/{samp_token}/fm_overall_{samp_token}.png', bbox_inches='tight')               
            # plt.imshow(fm)            
            plt.close(a)
            plt.close(fig)
            print('Atfer close :', process.memory_info().rss)  # in bytes
            del fig
            del a
            gc.collect()
            print('Atfer gc :', process.memory_info().rss)  # in bytes
            # seg, emb, dir = self.bevencode(topdown)
        
        #store imgs of 4 chns bev
        # for chn in range(seg.shape[1]):
        #     visualizer_plt(seg[0][chn], step=f"afterbevEncode_chn_{chn}")
        
        # segfm = torch.sum(seg.squeeze(0),0)
        # print(segfm.shape)
        # # 5 rows, 3 cols, index = imgcounter
        # a = fig.add_subplot(5,3,imgcounter)
        # plt.imshow(segfm.detach().cpu().numpy())
        # a.set_title('segfm')
        # plt.savefig(f'{imgs_path}/fm_overall_{samp_token}.png', bbox_inches='tight')  
              
        gc.collect()
        gc.collect()
        return self.bevencode(topdown) # torch.Size([1, 4, 200, 400])
