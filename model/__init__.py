from .hdmapnet import HDMapNet
from .ipm_net import IPMNet, TemporalIPMNet
from .lift_splat import LiftSplat
from .pointpillar import PointPillar

def get_model(method, data_conf, instance_seg=True, embedded_dim=16, direction_pred=True, angle_class=36):
    if method == 'lift_splat':
        model = LiftSplat(data_conf, instance_seg=instance_seg, embedded_dim=embedded_dim)
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
