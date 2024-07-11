from .hdmapnet import HDMapNet
from .ipm_net import IPMNet, TemporalIPMNet
from .lift_splat import LiftSplat
from .pointpillar import PointPillar

def get_model(method, data_conf, instance_seg=True, embedded_dim=16, direction_pred=True, angle_class=36):
    if method == 'lift_splat':
        viz_train=False
        nepochs=1
        H=900
        W=1600
        resize_lim=(0.193, 0.225)
        final_dim=(128, 352)
        bot_pct_lim=(0.0, 0.22)
        rot_lim=(-5.4, 5.4)
        rand_flip=True
        xbound=[-50.0, 50.0, 0.5]
        ybound=[-50.0, 50.0, 0.5]
        zbound=[-10.0, 10.0, 20.0]
        dbound=[4.0, 45.0, 1.0]
        bsz=1,
        nworkers=10        
        grid_conf = {
        'xbound': xbound,
        'ybound': ybound,
        'zbound': zbound,
        'dbound': dbound,
        }
        cams = ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT',
                             'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT']
        data_aug_conf = {
                    'resize_lim': resize_lim,
                    'final_dim': final_dim,
                    'rot_lim': rot_lim,
                    'H': H, 'W': W,
                    'rand_flip': rand_flip,
                    'bot_pct_lim': bot_pct_lim,
                    'cams': cams,
                    'Ncams': 5,
                }
        model = LiftSplat(grid_conf, data_aug_conf, outC=4, instance_seg=instance_seg, embedded_dim=embedded_dim)
    elif method == 'HDMapNet_cam':
#(self, data_conf, instance_seg=True, embedded_dim=16, direction_pred=True, direction_dim=36, lidar=False)
        model = HDMapNet(data_conf, instance_seg=instance_seg, embedded_dim=embedded_dim, direction_pred=direction_pred, direction_dim=angle_class, lidar=False)
    elif method == 'temporal_fusion':
# __init__(self, xbound, ybound, outC, camC=64, instance_seg=True, embedded_dim=16)
        model = TemporalIPMNet([-60,60,0.6], [-30,30,0.6], 4)
    elif method == 'HDMapNet_lidar':
        model = PointPillar(data_conf, embedded_dim=embedded_dim)
    elif method == 'HDMapNet_fusion':
        model = HDMapNet(data_conf, instance_seg=instance_seg, embedded_dim=embedded_dim, direction_pred=direction_pred, direction_dim=angle_class, lidar=True)
    else:
        raise NotImplementedError

    return model
