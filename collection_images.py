import os
import cv2
import shutil
import json
from PIL import Image

def read_json_file(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
        return data

def get_segment_file_path(sample_token):
    for root, dirs, files in os.walk(args.seg_path):
        for file in files:
            file_path = os.path.join(args.seg_path, file)
            if sample_token in file:
                return file_path

def get_vector_file_path(sample_token):
    for root, dirs, files in os.walk(args.vec_path):
        for file in files:
            file_path = os.path.join(args.vec_path, file)
            if sample_token in file:
                return file_path

def main(args):
    batchi = 0
    # cluster imge path
    cluster_path = os.path.join(args.output, 'image_cluster')
    if not os.path.exists(cluster_path):
        os.mkdir(cluster_path)

    samples = read_json_file(args.image_json)
    for smaple_token in samples:
        segment_img = get_segment_file_path(smaple_token)
        seg_img_cam_ori = Image.open(segment_img)
        seg_img_cam = seg_img_cam_ori.resize((2400, 1200))
        segment_sample_path = os.path.join(args.output, smaple_token)

        vector_img = get_vector_file_path(smaple_token)
        vec_img_cam_ori = Image.open(vector_img)
        vec_img_cam = vec_img_cam_ori.resize((2400, 1200))

        if not os.path.exists(segment_sample_path):
            os.mkdir(segment_sample_path)
        shutil.copy(segment_img, segment_sample_path)
        new_cam_image = Image.new('RGB', (1600*3, 900*2+1200))
        for img in samples[smaple_token]:
            camera = img['camera']
            filename = img['filename']
            img_cam = Image.open(filename)
            # img_cam = img_cam_ori.resize((800, 400))
            if camera == 'CAM_FRONT_LEFT':
                new_cam_image.paste(img_cam, (0, 0))
            elif camera == 'CAM_FRONT':
                new_cam_image.paste(img_cam, (1600, 0))
            elif camera == 'CAM_FRONT_RIGHT':
                new_cam_image.paste(img_cam, (3200, 0))
            elif camera == 'CAM_BACK_LEFT':
                new_cam_image.paste(img_cam, (0, 900))
            elif camera == 'CAM_BACK':
                new_cam_image.paste(img_cam, (1600, 900))
            elif camera == 'CAM_BACK_RIGHT':
                new_cam_image.paste(img_cam, (3200, 900))
            shutil.copy(filename, segment_sample_path)
        # new_cam_image.paste(seg_img_cam, (1200, 1800))
        new_cam_image.paste(seg_img_cam, (0, 1800))
        new_cam_image.paste(vec_img_cam, (2400, 1800))
        img_name = f'cluster_{batchi:06}_{smaple_token}.jpg'
        new_image_path = os.path.join(segment_sample_path, img_name)
        new_cam_image.save(new_image_path)
        shutil.copy(new_image_path, cluster_path)
        batchi = batchi + 1
    print(f"exported to {args.output}")

    # # the output vedio
    # output_video = os.path.join(args.output, 'image_cluster.mp4')
    # image_files = [f for f in sorted(os.listdir(cluster_path)) if f.endswith('.jpg')]
    # # 创建VideoWriter对象并指定编码器、分辨率等参数
    # fourcc = cv2.VideoWriter_fourcc(*'mp4v') # 选择合适的编码器（如XVID）
    # # frame_width = int(cv2.__version__.split(".")[0]) < 3 and 640 or 1920 # 根据OpenCV版本自动调整分辨率
    # # frame_height = int(cv2.__version__.split(".")[0]) < 3 and 480 or 1080
    # # out = cv2.VideoWriter(output_video, fourcc, 25.0, (frame_width, frame_height), True)
    # out = cv2.VideoWriter(output_video, fourcc, 5, (1600*3, 900*3-100))
    
    # for image_file in image_files:
    #     img = cv2.imread('{}/{}'.format(cluster_path, image_file))
    #     if img is None:
    #         print(image_file + " is error!")
    #         continue
    #     # 在此处进行其他操作或修改图像
    
    #     out.write(img) # 写入当前帧到视频文件
    # out.release() # 关闭视频文件
    # cv2.destroyAllWindows()
    # print("Done!")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Collect the images.')
    parser.add_argument('--seg_path', type=str, default='segment_result/')
    parser.add_argument('--vec_path', type=str, default='')
    parser.add_argument('--image_json', type=str, default='/samples.json')
    parser.add_argument('--output', type=str, default='/')
    args = parser.parse_args()

    main(args)
