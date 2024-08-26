import sys
 
if '/home/qichen/projects/HDMapNet-fusion' not in sys.path:
    sys.path.append('/home/qichen/projects/HDMapNet-fusion') 
print(sys.path)

from data.vector_map import VectorizedLocalMap
from data.dataset import *
from nuscenes.map_expansion.map_api import NuScenesMap, NuScenesMapExplorer

class mbmapdatatest(HDMapNetSemanticDataset):
    def __init__(self, version, dataroot, data_conf, is_train):
        super(mbmapdatatest, self).__init__(version, dataroot, data_conf, is_train)
        self.vector_map = mbmap(dataroot, patch_size=self.patch_size, canvas_size=self.canvas_size)

class mbmap(VectorizedLocalMap):
    def __init__(self,
                 dataroot,
                 patch_size,
                 canvas_size,
                 line_classes=['road_divider', 'lane_divider'],
                 ped_crossing_classes=['ped_crossing'],
                 contour_classes=['road_segment', 'lane'],
                 sample_dist=1,
                 num_samples=250,
                 padding=False,
                 normalize=False,
                 fixed_num=-1):
        super(mbmap, self).__init__(
                 dataroot,
                 patch_size,
                 canvas_size,
                 line_classes=['road_divider', 'lane_divider'],
                 ped_crossing_classes=['ped_crossing'],
                 contour_classes=['road_segment', 'lane'],
                 sample_dist=1,
                 num_samples=250,
                 padding=False,
                 normalize=False,
                 fixed_num=-1)
        '''
        Args:
            fixed_num = -1 : no fixed num
        '''
        self.MAPS = ['singapore-hollandvillage-mbfake']
        for loc in self.MAPS:
            self.nusc_maps[loc] = NuScenesMap(dataroot=self.data_root, map_name=loc)
            self.map_explorer[loc] = NuScenesMapExplorer(self.nusc_maps[loc])
        
def semantic_dataset(version, dataroot, data_conf, bsz, nworkers):
    train_dataset = mbmapdatatest(version='v1.0-trainval', dataroot='/home/qichen/projects/Nuscene-full/', data_conf=data_conf, is_train=True)
    val_dataset = mbmapdatatest(version='v1.0-trainval', dataroot='/home/qichen/projects/Nuscene-full/', data_conf=data_conf, is_train=False,)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=bsz, shuffle=True, num_workers=nworkers, drop_last=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=bsz, shuffle=False, num_workers=nworkers)
    return train_loader, val_loader

if __name__ == '__main__':
    data_conf = {
        'image_size': (900, 1600),
        'xbound': [-30.0, 30.0, 0.15],
        'ybound': [-15.0, 15.0, 0.15],
        'thickness': 5,
        'angle_class': 36,
    }

    dataset = mbmapdatatest(version='v1.0-trainval', dataroot='/home/qichen/projects/Nuscene-full/', data_conf=data_conf, is_train=False)
    for idx in range(len(dataset)):
        imgs, trans, rots, intrins, post_trans, post_rots, lidar_data, lidar_mask, car_trans, yaw_pitch_roll, semantic_masks, instance_masks, direction_mask, sample_token = dataset[idx]
