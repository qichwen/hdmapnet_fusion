"""
Script program
Author@Shiqi Jiang, xx.04.2024
"""

from PIL import Image, ImageFile
import os
import re
import numpy as np

MATERIAL_PATH = './material_images'
INTEGRATE_PATH = './integrate_images'
MAPTR_OUTPUT_PATH = '/home/sherlock/Pictures/mtr_outputs'

# 确保输出目录存在
if not os.path.exists(INTEGRATE_PATH):
    os.makedirs(INTEGRATE_PATH)

# 允许加载截断的图片文件
ImageFile.LOAD_TRUNCATED_IMAGES = True

def resize_and_rename_segment_images(base_folder_path):
    """
    遍历指定文件夹及其子文件夹，将名称开头为"segment"的jpg图像文件的宽高变为原来的四分之一，
    然后将文件名更改为以"segmentation"开头。

    Args:
        base_folder_path (str): 基础文件夹路径。
    """
    for root, dirs, files in os.walk(base_folder_path):
        for file in files:
            if file.startswith("segment_LSS") and file.endswith(".jpg"):
                old_file_path = os.path.join(root, file)
                
                try:
                    # 打开图像文件
                    with Image.open(old_file_path) as img:
                        # 计算新的尺寸
                        new_size = (img.width // 2, img.height // 2)
                        # 调整图像大小
                        resized_img = img.resize(new_size, Image.Resampling.LANCZOS)
                        # 保存调整后的图像到旧文件路径
                        resized_img.save(old_file_path)
                        print(f"Resized image: {old_file_path}")
                    
                    # 重命名文件
                    new_file_name = "segmentation" + file[len("segment"):]
                    new_file_path = os.path.join(root, new_file_name)
                    os.rename(old_file_path, new_file_path)
                    print(f"Renamed image: {old_file_path} -> {new_file_path}")
                except Exception as e:
                    print(f"Error processing image {old_file_path}: {e}")

def get_subfolders(directory):
    """
    一个生成器函数，遍历指定目录下的所有子文件夹。
    
    Args:
        directory (str): 需要遍历的目录路径。
    
    Yields:
        str: 子文件夹的完整路径。
    """
    for root, dirs, files in os.walk(directory):
        # 仅在顶层目录中查找子目录
        for name in dirs:
            yield os.path.join(root, name)
        break  # 防止深入子文件夹

def extract_min_number_from_folder(folder_path):
    """
    遍历指定文件夹中的所有图像文件，提取每个文件名中的六位整数，并返回最小的整数。
    
    Args:
        folder_path (str): 文件夹路径。
    
    Returns:
        int: 文件夹中所有图像文件名中的最小六位整数，如果没有找到任何整数，则返回None。
    """
    min_number = None
    for filename in os.listdir(folder_path):
        if filename.startswith('segmentation_') and filename.endswith('.jpg'):
            match = re.search(r'segmentation_(\d{6})_', filename)
            if match:
                number = int(match.group(1))
                if min_number is None or number < min_number:
                    min_number = number
    return min_number

def sort_folders_by_min_image_number(base_path):
    """
    根据子文件夹中图像文件名中的最小六位整数对子文件夹进行排序。
    
    Args:
        base_path (str): 总文件夹的路径。
    
    Returns:
        list: 排序后的子文件夹路径列表。
    """
    folders_with_numbers = []
    for folder_name in os.listdir(base_path):
        folder_path = os.path.join(base_path, folder_name)
        if os.path.isdir(folder_path):
            min_number = extract_min_number_from_folder(folder_path)
            if min_number is not None:
                folders_with_numbers.append((folder_path, min_number))

    # 根据每个文件夹中图像文件名的最小整数进行排序
    sorted_folders = [folder for folder, _ in sorted(folders_with_numbers, key=lambda x: x[1])]
    return sorted_folders

def rotate_and_rename_image(src_folders, dest_folders):
    """
    从源文件夹列表中提取图片，顺时针旋转90度，重命名，并保存到目标文件夹列表中。

    Args:
        src_folders (list): 源文件夹路径列表。
        dest_folders (list): 目标文件夹路径列表。
    """
    for src_folder, dest_folder in zip(src_folders, dest_folders):
        src_path = os.path.join(src_folder, "PRED_MAP_plot.png")
        dest_path = os.path.join(dest_folder, "segmentation_PRED_MAP_plot.png")
        
        # 检查源图片是否存在
        if os.path.exists(src_path):
            # 打开图片
            with Image.open(src_path) as img:
                # 顺时针旋转90度
                rotated_img = img.rotate(-90, expand=True)
                # 保存到目标路径
                rotated_img.save(dest_path)
                print(f"Image saved to {dest_path}")
        else:
            print(f"Image not found in {src_path}")

def image_integrater(MATERIAL_PATH, INTEGRATE_PATH):
    # 遍历MATERIAL_PATH下的每个子文件夹
    for subdir in os.listdir(MATERIAL_PATH):
        subdir_path = os.path.join(MATERIAL_PATH, subdir)
        if os.path.isdir(subdir_path):
            cam_images = []
            segmentation_images = []
            # 遍历子文件夹中的文件
            for file in sorted(os.listdir(subdir_path)):
                img_path = os.path.join(subdir_path, file)
                if file.endswith('.jpg'):
                    if 'pvimg_before_norm' in file:
                        cam_images.append((Image.open(img_path), file))
                    elif 'segmentation' in file:  # 匹配所有包含'segmentation'的图像文件
                        if file == "segmentation_PRED_MAP_plot.png":  # 特殊处理这个文件
                            with Image.open(img_path) as img:
                                # 计算新的宽度以保持宽高比
                                new_height = 200
                                aspect_ratio = img.width / img.height
                                new_width = int(new_height * aspect_ratio)
                                img_resized = img.resize((new_width, new_height), Image.ANTIALIAS)
                                segmentation_images.append(img_resized)
                        else:  # 其他'segmentation'图像正常添加
                            segmentation_images.append(Image.open(img_path))

            # 此处应当开始创建一个新图像，背景为白色
            new_image = Image.new('RGB', (1920, 1080), 'white')

            # 计算segmentation_images扩大后的高度
            seg_height = segmentation_images[0].height * 2 if segmentation_images else 0
            # 假设cam_images高度相同，计算两行的总高度
            cam_height = 2 * (cam_images[0][0].height if cam_images else 0)
            # 计算间隔（简单示例，可根据需要调整）
            gap = 50 if cam_images and segmentation_images else 0
            # 计算总高度
            total_height = seg_height + cam_height + gap
            
            # 确定起始y坐标以确保整体垂直居中
            start_y = (1080 - total_height) // 2
            
            # 粘贴segmentation_images
            # 计算segmentation_images扩大后的总宽度，包括间隙
            total_seg_width = sum(img.width * 2 for img in segmentation_images) + (len(segmentation_images) - 1) * 10

            # 确定起始x坐标以确保整体居中，包括间隙
            start_x_seg = (1920 - total_seg_width) // 2
            x_offset = start_x_seg
            for img in segmentation_images:
                seg_img_resized = img.resize((img.width * 2, img.height * 2))
                new_image.paste(seg_img_resized, (x_offset, start_y))
                x_offset += seg_img_resized.width + 10
            
            # 粘贴cam_images
            start_x_cam = (1920 - 3 * 496) // 2  # 假设每行最多3张图
            y_offset = start_y + seg_height + gap # 更新y坐标偏移量为segmentation_images的高度加上间隔
            for img, name in cam_images:
                first_char = name[0]
                row_offset = 0 if first_char in ['1', '2', '3'] else cam_images[0][0].height
                col = int(first_char) - 1 if first_char in ['1', '2', '3'] else abs(int(first_char) - 6) # int(first_char) - 4 or abs(int(first_char) - 6)
                new_image.paste(img, (start_x_cam + col * 496, y_offset + row_offset))
            
            # 保存合成图
            seg_img_name = segmentation_images[0].filename.split('/')[-1] if segmentation_images else 'default'
            output_filename = '_'.join(seg_img_name.split('_')[:-1]) + '.jpg'
            output_path = os.path.join(INTEGRATE_PATH, output_filename)
            new_image.save(output_path, format='JPEG')
            print(f"Saved integrated image to {output_path}")

def img_extracter_with_MapTR(MATERIAL_PATH, MAPTR_OUTPUT_PATH):
    subfolders = list(get_subfolders(MAPTR_OUTPUT_PATH))
    # 使用sorted()函数和自定义排序键来排序
    sorted_subfolders = sorted(subfolders, key=lambda x: (not 'n015' in x, x)) # MapTR outputs

    sorted_folders = sort_folders_by_min_image_number(MATERIAL_PATH) # HMN outputs

    rotate_and_rename_image(sorted_subfolders, sorted_folders)
    #print(sorted_subfolders)
    # for folder in sorted_folders:
    #     print(folder)
    

if __name__ == '__main__':
    # resize_and_rename_segment_images(MATERIAL_PATH)
    # img_extracter_with_MapTR(MATERIAL_PATH, MAPTR_OUTPUT_PATH)
    image_integrater(MATERIAL_PATH, INTEGRATE_PATH)