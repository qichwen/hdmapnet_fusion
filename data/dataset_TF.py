from data.dataset import *

class HDMapNetTemporalFusionDataset(HDMapNetSemanticDataset):
    def __init__(self, version, dataroot, data_conf, tf_size, is_train):
        super(HDMapNetTemporalFusionDataset, self).__init__(version, dataroot, data_conf, is_train)
        self.tf_size = tf_size
        # self.angle_class = data_conf['angle_class']
        
    def get_samples(self):
        samples = [samp for samp in self.nusc.sample]

        # remove samples that aren't in this split
        samples = [samp for samp in samples if
                self.nusc.get('scene', samp['scene_token'])['name'] in self.scenes]

        # group samples by scene
        scene_samples = {}
        for samp in samples:
            scene_token = samp['scene_token']
            if scene_token not in scene_samples:
                scene_samples[scene_token] = []
            scene_samples[scene_token].append(samp)

        # filter out scenes with less than 4 samples
        filtered_scene_samples = {scene: samples for scene, samples in scene_samples.items() if len(samples) > 4}

        # return the filtered samples
        return [samp for samples in filtered_scene_samples.values() for samp in samples]

    def get_semantic_map(self, rec):
        #Return list of map mask layers of the specified patch.
        vectors = self.get_vectors(rec)
        instance_masks, forward_masks, backward_masks = preprocess_map(vectors, self.patch_size, self.canvas_size, NUM_CLASSES, self.thickness, self.angle_class)
        semantic_masks = instance_masks != 0
        semantic_masks = torch.cat([(~torch.any(semantic_masks, axis=0)).unsqueeze(0), semantic_masks])
        instance_masks = instance_masks.sum(0)
        forward_oh_masks = label_onehot_encoding(forward_masks, self.angle_class+1)
        backward_oh_masks = label_onehot_encoding(backward_masks, self.angle_class+1)
        direction_masks = forward_oh_masks + backward_oh_masks
        direction_masks = direction_masks / direction_masks.sum(0)
        return semantic_masks, instance_masks, forward_masks, backward_masks, direction_masks

    def __getitem__(self, idx):
        """
        window = []
        for i in range(len(frames) - window_size + 1):
            window_frames = frames[i:i + window_size]
            window_frames = torch.stack([frame['image'] for frame in window_frames])
            window.append(window_frames)
        return torch.stack(window)
        """
        from copy import deepcopy
        rec_tf = []                
        imgs, rots, trans, intrins, post_rots, post_trans, car_trans, yaw_pitch_roll, sample_token = [], [], [], [], [], [], [], [], []

        # Collect data for each frame
        for i in range(0, self.tf_size):
            #TODO: controll to pick only sample from self.samples via [4:], corner cases?
            if (idx + self.tf_size - i) > len(self.samples): 
                print("out of range of the len-samples")
                break 
            rec = self.samples[idx + self.tf_size - i]  
            _imgs, _trans, _rots, _intrins, _post_trans, _post_rots = self.get_imgs(rec) 
            # torch.stack(imgs), torch.stack(trans), torch.stack(rots), torch.stack(intrins), torch.stack(post_trans), torch.stack(post_rots)
            _car_trans, _car_ypr = self.get_ego_pose(rec)
            sample_token.append(rec.get('token'))

            imgs.append(_imgs)
            rots.append(_rots)
            trans.append(_trans)
            intrins.append(_intrins)
            post_rots.append(_post_rots)
            post_trans.append(_post_trans)
            car_trans.append(_car_trans)
            yaw_pitch_roll.append(_car_ypr)

        # Convert lists to tensors
        imgs = torch.stack(imgs)
        rots = torch.stack(rots)
        trans = torch.stack(trans)
        intrins = torch.stack(intrins)
        post_rots = torch.stack(post_rots)
        post_trans = torch.stack(post_trans)
        car_trans = torch.stack(car_trans)
        yaw_pitch_roll = torch.stack(yaw_pitch_roll)
        # sample_token = torch.tensor(sample_token)

        # Reshape imgs tensor to (B, T, N, C, imH, imW)
        # imgs = imgs.permute(1, 0, 2, 3, 4, 5)

        return imgs, rots, trans, intrins, post_rots, post_trans, car_trans, yaw_pitch_roll, sample_token                     
        # Needed x, rots, trans, intrins, post_rots, post_trans, translation, yaw_pitch_roll,smp

def dataset_tf(version, dataroot, data_conf, bsz, nworkers, tf_size=4):
    #train_dataset = HDMapNetSemanticDataset(version, dataroot, data_conf, is_train=True)
    # val_dataset = HDMapNetSemanticDataset(version, dataroot, data_conf, is_train=False)
    
    # train_dataset_tf = HDMapNetTemporalFusionDataset(version, dataroot, data_conf, tf_size, is_train=False, )
    val_dataset_tf = HDMapNetTemporalFusionDataset(version, dataroot, data_conf, tf_size, is_train=False)
    
    # Partition data into batches
    #train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=bsz, shuffle=True, num_workers=nworkers, drop_last=True)
    # val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=bsz, shuffle=False, num_workers=nworkers)
    
    # Temporal dimension construction for trainning
    #train_loader = torch.utils.data.DataLoader(train_dataset_tf, batch_size=bsz, shuffle=True, num_workers=nworkers, drop_last=True)
    val_loader_tf = torch.utils.data.DataLoader(val_dataset_tf, batch_size=bsz, shuffle=False, num_workers=nworkers)


    # print(val_loader)
    
    #return train_loader, val_loader_tf
    
    #tf
    return val_loader_tf


if __name__ == '__main__':
    data_conf = {
        'image_size': (900, 1600),
        'xbound': [-30.0, 30.0, 0.15],
        'ybound': [-15.0, 15.0, 0.15],
        'thickness': 5,
        'angle_class': 36,
    }

    dataset = HDMapNetSemanticDataset(version='v1.0-mini', dataroot='dataset/nuScenes', data_conf=data_conf, is_train=False)
    for idx in range(dataset.__len__()):
        imgs, trans, rots, intrins, post_trans, post_rots, lidar_data, lidar_mask, car_trans, yaw_pitch_roll, semantic_masks, instance_masks, direction_mask = dataset.__getitem__(idx)

