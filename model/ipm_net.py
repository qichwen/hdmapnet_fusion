import torch
from torch import nn

from .homography import IPM, bilinear_sampler
from .utils import plane_grid_2d, get_rot_2d, cam_to_pixel
from .base import CamEncode, BevEncode
import matplotlib.pyplot as plt
import os
from image_processer import *
from data.const import *
import gc

class IPMNet(nn.Module):
    def __init__(self, xbound, ybound, outC, camC=64, instance_seg=True, embedded_dim=16, cam_encoding=True, bev_encoding=True, z_roll_pitch=False):
        super(IPMNet, self).__init__()
        self.xbound = xbound
        self.ybound = ybound
        self.camC = camC
        self.downsample = 16
        if cam_encoding:
            self.ipm = IPM(xbound, ybound, N=6, C=camC, visual=True, z_roll_pitch=z_roll_pitch, extrinsic=False)
        else:
            self.ipm = IPM(xbound, ybound, N=6, C=camC, visual=True, z_roll_pitch=z_roll_pitch, extrinsic=False)
        self.cam_encoding = cam_encoding
        if cam_encoding:
            self.camencode = CamEncode(camC)
        self.bev_encoding = bev_encoding
        if bev_encoding:
            self.bevencode = BevEncode(inC=camC, outC=outC, instance_seg=instance_seg, embedded_dim=embedded_dim)

    def get_cam_feats(self, x):
        """Return B x N x D x H/downsample x W/downsample x C
        """
        B, N, C, imH, imW = x.shape

        x = x.view(B*N, C, imH, imW)
        x = self.camencode(x)
        x = x.view(B, N, self.camC, imH//self.downsample, imW//self.downsample)
        return x

    def get_Ks_RTs_and_post_RTs(self, intrins, rots, trans, post_rots, post_trans):
        B, N, _, _ = intrins.shape
        Ks = torch.eye(4, device=intrins.device).view(1, 1, 4, 4).repeat(B, N, 1, 1)
        Ks[:, :, :3, :3] = intrins

        Rs = torch.eye(4, device=rots.device).view(1, 1, 4, 4).repeat(B, N, 1, 1)
        Rs[:, :, :3, :3] = rots.transpose(-1, -2).contiguous()
        Ts = torch.eye(4, device=trans.device).view(1, 1, 4, 4).repeat(B, N, 1, 1)
        Ts[:, :, :3, 3] = -trans
        RTs = Rs @ Ts

        post_RTs = torch.eye(4, device=post_rots.device).view(1, 1, 4, 4).repeat(B, N, 1, 1)
        post_RTs[:, :, :3, :3] = post_rots
        post_RTs[:, :, :3, 3] = post_trans

        if self.cam_encoding:
            scale = torch.Tensor([
                [1/self.downsample, 0, 0, 0],
                [0, 1/self.downsample, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1]
            ]).cuda()
            post_RTs = scale @ post_RTs

        return Ks, RTs, post_RTs

    def forward(self, points, points_mask, x, rots, trans, intrins, post_rots, post_trans, translation, yaw_pitch_roll):
        if self.cam_encoding:
            x = self.get_cam_feats(x)

        Ks, RTs, post_RTs = self.get_Ks_RTs_and_post_RTs(intrins, rots, trans, post_rots, post_trans)
        topdown = self.ipm(x, Ks, RTs, translation, yaw_pitch_roll, post_RTs)

        if self.bev_encoding:
            return self.bevencode(topdown)
        else:
            return topdown
        
class TemporalIPMNet(IPMNet):
    def __init__(self, xbound, ybound, outC, camC=64, instance_seg=True, embedded_dim=16):
        super(TemporalIPMNet, self).__init__(xbound, ybound, outC, camC, instance_seg, embedded_dim, cam_encoding=True, bev_encoding=True, z_roll_pitch=False)

    def get_cam_feats(self, x):
        """Return B x T x N x H/downsample x W/downsample x C
        """
        B, T, N, C, imH, imW = x.shape
        """
        B: Batch size. This is the number of samples or instances in the batch.
        T: Time or sequence length. This dimension often represents a sequence of frames, time steps, or other sequential data.
        N: Number of objects, keypoints, or features. This dimension might represent a set of objects, keypoints, or features extracted from an image or video.
        C: Number of channels or features. This dimension typically represents the number of color channels (e.g., RGB) or feature channels (e.g., depth, normals) in an image.
        imH: Image height. This is the vertical resolution of the image.
        imW: Image width. This is the horizontal resolution of the image.
        """
        x = x.view(B*T*N, C, imH, imW)
        x = self.camencode(x)
        x = x.view(B, T, N, self.camC, imH//self.downsample, imW//self.downsample)
        return x

    def temporal_fusion(self, topdown, translation, yaw, smp):
        #
        draw_tensor(smp, topdown)
        
        B, T, C, H, W = topdown.shape   
        
        if T == 1:
            
            return topdown[:, 0]

        grid = plane_grid_2d(self.xbound, self.ybound).view(1, 1, 2, H*W).repeat(B, T-1, 1, 1)
        rot0 = get_rot_2d(yaw[:, 1:])
        trans0 = translation[:, 1:, :2].view(B, T-1, 2, 1)
        rot1 = get_rot_2d(yaw[:, 0].view(B, 1).repeat(1, T-1))
        trans1 = translation[:, 0, :2].view(B, 1, 2, 1).repeat(1, T-1, 1, 1)
        grid = rot1.transpose(2, 3) @ grid
        grid = grid + trans1
        grid = grid - trans0
        grid = rot0 @ grid
        grid = grid.view(B*(T-1), 2, H, W).permute(0, 2, 3, 1).contiguous()
        grid = cam_to_pixel(grid, self.xbound, self.ybound)
        topdown = topdown.permute(0, 1, 3, 4, 2).contiguous()
        prev_topdown = topdown[:, 1:]
        warped_prev_topdown = bilinear_sampler(prev_topdown.reshape(B*(T-1), H, W, C), grid).view(B, T-1, H, W, C)
        topdown = torch.cat([topdown[:, 0].unsqueeze(1), warped_prev_topdown], axis=1)
        topdown = topdown.view(B, T, H, W, C)
        topdown = topdown.max(1)[0]
        topdown = topdown.permute(0, 3, 1, 2).contiguous()
        return topdown

    # def forward(self, points, points_mask, x, rots, trans, intrins, post_rots, post_trans, translation, yaw_pitch_roll,smp):
    def forward(self, x, rots, trans, intrins, post_rots, post_trans, translation, yaw_pitch_roll,smp):
        x = self.get_cam_feats(x)
        
        B, T, N, C, h, w = x.shape
        x = x.view(B*T, N, C, h, w)
        
        # x,counter, sample_token, sample_channel, ts, step
        for i in range(x.size(1)):
            visualizer_plt(x[0][i].squeeze(0),i, smp[0][0], CAMS[i] , '0', '0', 'plt_images/tempfusion/')
        
        intrins = intrins.view(B*T, N, 3, 3)
        rots = rots.view(B*T, N, 3, 3)
        trans = trans.view(B*T, N, 3)
        post_rots = post_rots.view(B*T, N, 3, 3)
        post_trans = post_trans.view(B*T, N, 3)
        
        Ks, RTs, post_RTs = self.get_Ks_RTs_and_post_RTs(intrins, rots, trans, post_rots, post_trans)
        #TODO: something wrong here in IPM
        topdown = self.ipm(x, Ks, RTs, smp[0][0], translation, yaw_pitch_roll, post_RTs)
        _, C, H, W = topdown.shape
        topdown = topdown.view(B, T, C, H, W)
        
        topdown = self.temporal_fusion(topdown, translation, yaw_pitch_roll[..., 0], smp)
        # if draw:
        
        # seg, emb, dir = self.bevencode(topdown)
        # #seg : (1,4, 200, 400)
        # segfm = torch.sum(seg.squeeze(0),0)
        # print(segfm.shape)
        # # 5 rows, 3 cols, index = imgcounter
        # loc = imgcounter + 1      
        # a = fig.add_subplot(rows, cols, loc)
        # # a.axis('off')
        # # plt.axis('off')
        # plt.imshow(segfm.detach().cpu().numpy())
        # a.set_title('segfm_BEV_topdown')
        # imgcounter += 1          
                    
        # plt.savefig(f'plt_images/{samp_token}/fm_overall_{samp_token}.png', bbox_inches='tight')               
        # # plt.imshow(fm)            
        # plt.close(a)
        # plt.close(fig)
        # print('Atfer close :', process.memory_info().rss)  # in bytes
        # del fig
        # del a
        # gc.collect()
        # print('Atfer gc :', process.memory_info().rss)  # in bytes
        
        
        return self.bevencode(topdown)

        # Original hdmapnet 
        # def forward(self, img, trans, rots, intrins, post_trans, post_rots, lidar_data, lidar_mask, car_trans, yaw_pitch_roll):
        # x = self.get_cam_feats(img)
        # x = self.view_fusion(x)
        # Ks, RTs, post_RTs = self.get_Ks_RTs_and_post_RTs(intrins, rots, trans, post_rots, post_trans)
        # topdown = self.ipm(x, Ks, RTs, car_trans, yaw_pitch_roll, post_RTs)
        # topdown = self.up_sampler(topdown)
        # if self.lidar:
        #     lidar_feature = self.pp(lidar_data, lidar_mask)
        #     topdown = torch.cat([topdown, lidar_feature], dim=1)
        # return self.bevencode(topdown)