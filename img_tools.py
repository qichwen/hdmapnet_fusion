import cv2
import numpy as np
import torch
import torchvision.transforms as transforms

def isp_module(image):
    """
    模拟ISP处理，包括白平衡、曝光调整、降噪和锐化。

    参数:
        image (numpy.ndarray): 输入的BGR图像。

    返回:
        numpy.ndarray: 处理后的图像。
    """
    
    # 白平衡
    def simple_white_balance(img):
        """
        简单的白平衡调整。
        """
        # 分离三个颜色通道
        b, g, r = cv2.split(img)
        
        # 获取每个通道的平均值
        b_avg, g_avg, r_avg = map(np.mean, [b, g, r])
        
        # 获取所有通道的平均值
        avg = (b_avg + g_avg + r_avg) / 3
        
        # 计算每个通道的增益系数
        b_gain = avg / b_avg
        g_gain = avg / g_avg
        r_gain = avg / r_avg
        
        # 应用增益系数
        b = cv2.multiply(b, b_gain)
        g = cv2.multiply(g, g_gain)
        r = cv2.multiply(r, r_gain)
        
        # 合并调整后的通道
        return cv2.merge([b, g, r])
    
    def white_balance2(img):
        # Convert the image to LAB color space
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        
        # Split the LAB image into its channels
        l, a, b = cv2.split(lab)
        
        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to the L-channel
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        
        # Merge the CLAHE enhanced L-channel back with A and B channels
        limg = cv2.merge((cl, a, b))
        
        # Convert the LAB image back to BGR color space
        balanced_img = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
        
        return balanced_img

    # 曝光调整
    def adjust_exposure(img):
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        v = cv2.equalizeHist(v)
        final_hsv = cv2.merge((h, s, v))
        img_result = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
        return img_result

    # 降噪
    def denoise(img):
        result = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)
        return result
    
    def color_adaptation(img, saturation_scale=1.3):
        """
        调整图像的饱和度以增强色彩表现。
        
        参数:
            img (numpy.ndarray): 输入的BGR图像。
            saturation_scale (float): 饱和度调整因子，默认值为1.3。
            
        返回:
            numpy.ndarray: 色彩自适应后的图像。
        """
        # 将图像从BGR转换到HSV
        hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv_img)
        
        # 调整饱和度
        s = cv2.multiply(s, saturation_scale)
        s = np.clip(s, 0, 255)  # 确保饱和度值在合法范围内
        
        # 重新组合并转换回BGR
        adjusted_hsv = cv2.merge([h, s, v])
        adjusted_img = cv2.cvtColor(adjusted_hsv, cv2.COLOR_HSV2BGR)
        return adjusted_img

    # 锐化
    def sharpen(img):
        kernel = np.array([[-1, -1, -1],
                           [-1,  9, -1],
                           [-1, -1, -1]])
        result = cv2.filter2D(img, -1, kernel)
        return result

    # 应用白平衡
    # wb_img = simple_white_balance(image)
    wb_img = white_balance2(image)
    
    # 应用曝光调整
    # exp_img = adjust_exposure(wb_img)
    
    # 应用降噪
    # dn_img = denoise(wb_img)

    # 应用色彩自适应
    ca_img = color_adaptation(wb_img)
    
    # 应用锐化
    # sharp_img = sharpen(wb_img)
    
    return ca_img

def gaussian_blur(img):
    # 定义高斯核
    kernel = np.array([[1, 2, 1], 
                       [2, 4, 2], 
                       [1, 2, 1]], np.float32) / 16
    
    kernel2 = np.array([[1, 4, 7, 4, 1],
                       [4, 16, 26, 16, 4], 
                       [7, 26, 41, 26, 7],
                       [4, 16, 26, 16, 4],
                       [1, 4, 7, 4, 1]], np.float32) / 273
    
    # 应用高斯滤波
    gaussian_img = cv2.filter2D(img, -1, kernel)
    return gaussian_img

def histogram_equalization(img):
    img_yuv = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
    img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])
    img_equalized = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2RGB)
    return img_equalized

def median_filter(image, kernel_size=3):
    """
    对图像应用中值滤波。

    :param input_image_path: 输入图像的文件路径
    :param output_image_path: 输出图像的文件路径
    :param kernel_size: 中值滤波的核大小，必须是奇数
    """
    # 读取图像
    # image = cv2.imread(input_image_path)
    if image is None:
        print(f"Failed to read image: {image}")
        return
    # 应用中值滤波
    filtered_image = cv2.medianBlur(image, kernel_size)

    return filtered_image

def white_balance(img):
    # Convert the image to LAB color space
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    
    # Split the LAB image into its channels
    l, a, b = cv2.split(lab)
    
    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to the L-channel
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    
    # Merge the CLAHE enhanced L-channel back with A and B channels
    limg = cv2.merge((cl, a, b))
    
    # Convert the LAB image back to BGR color space
    balanced_img = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    
    return balanced_img

def colour_norm(img):
    dst = np.zeros_like(img)
    img_merge = cv2.normalize(img, dst, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U) 

    return img_merge

def clahe(img):
    img_yuv = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(3, 3))
    img_yuv[:, :, 0] = clahe.apply(img_yuv[:, :, 0])
    img_clahe = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2RGB)
    return img_clahe

def adjust_brightness_contrast(img, alpha= 1, beta= 65):
    enhanced_img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
    return enhanced_img

def feature_enhance(image, lower_rgb_w=[90, 80, 73], upper_rgb_w=[130, 120, 100], new_rgb_w=[100, 120, 150], 
                    lower_rgb_y=[90, 70, 50], upper_rgb_y=[120, 100, 70], new_rgb_y=[250, 250, 200]):
    """
    改变图像中指定RGB区间的像素值
    仅对图像的下半部分进行处理
    
    Parameters:
    - image: 输入的图像 (numpy array)
    - lower_rgb_w: 白色特征的低阈值RGB值 (tuple/list of 3 integers)
    - upper_rgb_w: 白色特征的高阈值RGB值 (tuple/list of 3 integers)
    - new_rgb_w: 白色特征的新RGB值 (tuple/list of 3 integers)
    - lower_rgb_y: 黄色特征的低阈值RGB值 (tuple/list of 3 integers)
    - upper_rgb_y: 黄色特征的高阈值RGB值 (tuple/list of 3 integers)
    - new_rgb_y: 黄色特征的新RGB值 (tuple/list of 3 integers)
    
    Returns:
    - 输出的图像 (numpy array)
    """
    # 获取图像的高度和宽度
    height, width = image.shape[:2]
    
    # 提取图像的下半部分
    lower_half = image[height//2:, :]
    
    # 创建掩码，找到在指定RGB区间内的像素
    mask_w = cv2.inRange(lower_half, np.array(lower_rgb_w), np.array(upper_rgb_w))
    mask_y = cv2.inRange(lower_half, np.array(lower_rgb_y), np.array(upper_rgb_y))
    
    # 将掩码应用到下半部分图像上，并将符合条件的像素值更改为新的RGB值
    lower_half[mask_w != 0] = new_rgb_w
    lower_half[mask_y != 0] = new_rgb_y
    
    # 将增强后的下半部分与上半部分重新合并
    image[height//2:, :] = lower_half
    
    return image

def mask(image, threshold=150, value=0):
    """
    将图像中灰度值大于阈值的像素的灰度值设置为指定的变量值，并返回处理后的 RGB 图像。

    Args:
        image (numpy.ndarray): 输入的 RGB 图像。
        threshold (int): 灰度阈值。
        value (int): 替换后的灰度值。

    Returns:
        numpy.ndarray: 处理后的 RGB 图像。
    """
    # 确保图像是 RGB 格式
    if len(image.shape) != 3 or image.shape[2] != 3:
        raise ValueError("Input image must be an RGB image.")
    
    # 将图像转换为灰度图像
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    # 创建一个掩码，灰度值大于阈值的像素设置为 True
    mask = gray_image > threshold
    
    # 创建一个新的灰度图像，将满足条件的像素值设置为指定的变量值
    modified_gray_image = gray_image.copy()
    modified_gray_image[mask] = value
    
    # 将修改后的灰度图像与原始图像的颜色信息结合
    modified_rgb_image = image.copy()
    for i in range(3):  # 对每个通道进行处理
        modified_rgb_image[:, :, i][mask] = value
    
    return modified_rgb_image

def normalize_hdm_noBlackPixel(img, mean, std):
    """
    只对图像中非黑色像素值 (RGB: 0, 0, 0) 的区域进行归一化。

    Args:
        img (torch.Tensor): 输入的 RGB 图像张量。
        mean (list): 每个通道的均值。
        std (list): 每个通道的标准差。

    Returns:
        torch.Tensor: 归一化后的图像张量。
    """
    # 创建一个掩码，标识非黑色像素
    mask = torch.any(img != 0, dim=0)

    # 创建一个归一化后的张量
    normalized_img = img.clone()

    # 定义归一化参数
    mean = torch.tensor(mean).view(3, 1, 1)
    std = torch.tensor(std).view(3, 1, 1)

    # 仅对非黑色像素进行归一化
    for c in range(3):  # 对每个通道进行处理
        channel = img[c, :, :]
        non_black_pixels = channel[mask]
        if len(non_black_pixels) > 0:  # 确保有非黑色像素
            normalized_channel = (channel - mean[c]) / std[c]
            normalized_img[c, :, :][mask] = normalized_channel[mask]

    return normalized_img

def denormalize_hdm_noBlackPixel(normalized_img, mean, std):
    """
    只对图像中非黑色像素值 (RGB: 0, 0, 0) 的区域进行反归一化。

    Args:
        normalized_img (torch.Tensor): 归一化后的 RGB 图像张量。
        mean (list): 每个通道的均值。
        std (list): 每个通道的标准差。

    Returns:
        torch.Tensor: 反归一化后的图像张量。
    """
    # 创建一个掩码，标识非黑色像素
    mask = torch.any(normalized_img != 0, dim=0)

    # 创建一个反归一化后的张量
    denormalized_img = normalized_img.clone()

    # 定义反归一化参数
    mean = torch.tensor(mean).view(3, 1, 1)
    std = torch.tensor(std).view(3, 1, 1)

    # 仅对非黑色像素进行反归一化
    for c in range(3):  # 对每个通道进行处理
        channel = normalized_img[c, :, :]
        non_black_pixels = channel[mask]
        if len(non_black_pixels) > 0:  # 确保有非黑色像素
            denormalized_channel = channel * std[c] + mean[c]
            denormalized_img[c, :, :][mask] = denormalized_channel[mask]

    return denormalized_img

#########################################################################################################################

def normalization_viaCV2(image_rgb):
    # 归一化RGB图像
    normalized_image_rgb = np.zeros_like(image_rgb)
    for i in range(3):  # 对每个通道进行归一化
        normalized_image_rgb[:, :, i] = cv2.normalize(image_rgb[:, :, i], None, 0, 255, cv2.NORM_MINMAX)
    
    # # 将图像转换为灰度图像
    # gray_image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    
    # # 归一化灰度图像
    # normalized_image_grey = cv2.normalize(gray_image, None, 0, 255, cv2.NORM_MINMAX)
    
    return normalized_image_rgb

def normalization_viaCV2_noBlackPixel(image_rgb):
    """
    只对图像中非黑色像素值 (RGB: 0, 0, 0) 的区域进行归一化。

    Args:
        image_rgb (numpy.ndarray): 输入的 RGB 图像。

    Returns:
        numpy.ndarray: 归一化后的 RGB 图像。
    """
    # 创建一个掩码，标识非黑色像素
    mask = np.any(image_rgb != [0, 0, 0], axis=-1)

    # 归一化 RGB 图像
    normalized_image_rgb = image_rgb.copy()
    for i in range(3):  # 对每个通道进行归一化
        channel = image_rgb[:, :, i]
        # 仅对非黑色像素进行归一化
        non_black_pixels = channel[mask]
        if len(non_black_pixels) > 0:  # 确保有非黑色像素
            min_val = non_black_pixels.min()
            max_val = non_black_pixels.max()
            if max_val > min_val:  # 避免除以零
                normalized_channel = (channel - min_val) / (max_val - min_val) * 255
                normalized_image_rgb[:, :, i][mask] = normalized_channel[mask]
    
    return normalized_image_rgb

def normalization_viaTorch(image_rgb):
    # 将图像从 numpy 数组转换为 PyTorch 张量
    image_tensor = torch.tensor(image_rgb, dtype=torch.float32)
    
    # 归一化 RGB 图像
    normalized_tensor_rgb = torch.zeros_like(image_tensor)
    for i in range(3):  # 对每个通道进行归一化
        channel = image_tensor[:, :, i]
        min_val = channel.min()
        max_val = channel.max()
        normalized_tensor_rgb[:, :, i] = (channel - min_val) / (max_val - min_val) * 255
    
    # 将 RGB 图像转换为灰度图像
    # gray_tensor = 0.2989 * image_tensor[:, :, 0] + 0.5870 * image_tensor[:, :, 1] + 0.1140 * image_tensor[:, :, 2]
    
    # 归一化灰度图像
    # min_val = gray_tensor.min()
    # max_val = gray_tensor.max()
    # normalized_gray_tensor = (gray_tensor - min_val) / (max_val - min_val) * 255
    
    # 将归一化后的张量转换回 numpy 数组
    normalized_image_rgb = normalized_tensor_rgb.numpy().astype(np.uint8)
    # normalized_gray_image = normalized_gray_tensor.numpy().astype(np.uint8)
    
    return normalized_image_rgb

#######################################################################################################################
# import os
# from PIL import Image
# import matplotlib.pyplot as plt

# def display_image_and_histogram(image_path):
#     # 打开图像
#     img = Image.open(image_path)
    
#     # 将图像转换为 RGB 模式
#     img = img.convert('RGB')
    
#     # 提取 RGB 通道数据
#     r, g, b = img.split()
#     r_data = r.getdata()
#     g_data = g.getdata()
#     b_data = b.getdata()
    
#     # 显示原图像
#     plt.figure(figsize=(12, 6))
    
#     plt.subplot(1, 2, 1)
#     plt.imshow(img)
#     plt.title('Original Image')
#     plt.axis('off')
    
#     # 计算并显示 RGB 通道的直方图
#     plt.subplot(1, 2, 2)
#     plt.hist(r_data, bins=256, color='red', alpha=0.6, label='Red Channel')
#     plt.hist(g_data, bins=256, color='green', alpha=0.6, label='Green Channel')
#     plt.hist(b_data, bins=256, color='blue', alpha=0.6, label='Blue Channel')
#     plt.title('RGB Channel Histogram')
#     plt.xlabel('Pixel Intensity')
#     plt.ylabel('Frequency')
#     plt.legend()
#     plt.xlim([0, 256])
    
#     # 显示图像和直方图
#     plt.tight_layout()
#     plt.show()

# # 输入图像文件路径
# # image_path_n005_01_back = '/home/sherlock/Pictures/RGB_test_platform/n005/01-21/01_3RKUTmMz98TDI9Rdq4CPg8JJw30gEZTc/5_pvimg_before_norm_CAM_BACK_5408.9.jpg'
# image_path_n005_21_front = '/home/sherlock/Pictures/RGB_test_platform/n005/01-21-41/21_TD8h088sp1fabZRtnBCLAaauQdMFqbqm/2_pvimg_before_norm_CAM_FRONT_5418.9.jpg'
# # image_path_n005_01_front = '/home/sherlock/Pictures/RGB_test_platform/n005/01-21-41/21_TD8h088sp1fabZRtnBCLAaauQdMFqbqm/2_pvimg_before_norm_CAM_FRONT_5418.9.jpg'

# image_path_n005_21_front_brightUP = '/home/sherlock/Pictures/RGB_test_platform/n005/01-21-41/21_brightUP_TD8h088sp1fabZRtnBCLAaauQdMFqbqm/2_pvimg_before_norm_CAM_FRONT_5418.9.jpg'


# # image_path_n005_21_back = '/home/sherlock/Pictures/RGB_test_platform/n005/01-21/21_TD8h088sp1fabZRtnBCLAaauQdMFqbqm/5_pvimg_before_norm_CAM_BACK_5418.9.jpg'
# # image_path_n005_41_back = '/home/sherlock/Pictures/RGB_test_platform/n005/01-21/41_Z1xpikK3iSMFNNroyrMCUY7HKawkCHoP/5_pvimg_before_norm_CAM_BACK_5428.9.jpg'

# # 调用函数显示图像及其 RGB 直方图
# # display_image_and_histogram(image_path_n005_01_back)
# display_image_and_histogram(image_path_n005_21_front)
# display_image_and_histogram(image_path_n005_21_front_brightUP)
# # display_image_and_histogram(image_path_n005_41_back)