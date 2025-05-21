from PIL import Image, PngImagePlugin, JpegImagePlugin
import torch
import cv2
import os
import numpy as np

def add_filter_whoo(args,image_file):
    # print(image_file)
    images=[]
    if 'gussian' in args.filter_type:
        # print('1211')
        image = np.array(image_file)
        # print('1211')
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  
        # print('1211')
        # image = cv2.GaussianBlur(image, (15, 15), 10)
        # image = Image.fromarray(image) 
        for i in range(50,200,50):
            # print('1211')
            gauss_image = cv2.GaussianBlur(image, (15, 15), i)
            gauss_image = Image.fromarray(gauss_image) 
            images.append(gauss_image)
            # print('1211')
    return images

def add_filter_adjust_sigma(args, image_file, sigma, image_processor):
    images=[]
    passing = 1 if isinstance(image_file, PngImagePlugin.PngImageFile) or isinstance(image_file, JpegImagePlugin.JpegImageFile) or isinstance(image_file, Image.Image) else 0
    if 'gussian' in args.filter_type:
        if passing:
            image = np.array(image_file)
        else:
            image = cv2.imread(os.path.join(args.image_folder,image_file))  
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) 
        # image = cv2.GaussianBlur(image, (15, 15), 10)
        # image = Image.fromarray(image) 
        gauss_image = cv2.GaussianBlur(image, (15, 15), sigma)
        gauss_image = Image.fromarray(image) 
        images.append(gauss_image)
            
    elif 'median' in args.filter_type:
        if passing:
            image = np.array(image_file)
        else:
            image = cv2.imread(os.path.join(args.image_folder, image_file))  
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        median_image = cv2.medianBlur(image, sigma)
        median_image = Image.fromarray(median_image)
        images.append(median_image)
    
    # To be modified
    # elif 'sharpen' in args.filter_type:
    #     image = cv2.imread(os.path.join(args.image_folder,image_file))  
    #     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  
    #     for intensity in range(3, 6):  # 중심값을 3에서 5로 변경
    #         # 두 번째 루프: 커널 크기 조정
    #         for kernel_size in range(3, 6, 2):  # 3x3과 5x5 크기 커널 사용
    #             sharpen_kernel = np.array([[0, -1, 0], [-1, intensity, -1], [0, -1, 0]])
                
    #             # 세 번째 루프: 반복 적용 횟수 조정
    #             sharpen_image = image.copy()
    #             for _ in range(1, 4):  # 필터를 최대 3회 반복 적용
    #                 sharpen_image = cv2.filter2D(sharpen_image, -1, sharpen_kernel)
                
    #             sharpen_image = Image.fromarray(sharpen_image)
    #             images.append(sharpen_image)
    elif 'sharpen' in args.filter_type:
        if passing:
            image = np.array(image_file)
        else:
            image = cv2.imread(os.path.join(args.image_folder,image_file))  
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  
        b = (1 - sigma) / 8
        sharpening_kernel = np.array([[b, b, b],
                                    [b, sigma, b],
                                    [b, b, b]])
        sharpened_image = cv2.filter2D(image, -1, sharpening_kernel)
        sharpened_image = Image.fromarray(sharpened_image)
        images.append(sharpened_image)
        
    elif 'sobel' in args.filter_type:
        if passing:
            image = np.array(image_file)
        else:        
            image = cv2.imread(os.path.join(args.image_folder,image_file))  
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  
        # Sobel edge detection
        sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=sigma)
        sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=sigma)
        sobel_image = cv2.magnitude(sobel_x, sobel_y)
        sobel_image = cv2.convertScaleAbs(sobel_image)
        sobel_image = Image.fromarray(sobel_image)
        images.append(sobel_image)
        
    elif 'bilateral' in args.filter_type:
        if passing:
            image = np.array(image_file)
        else:
            image = cv2.imread(os.path.join(args.image_folder,image_file))  
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  
        bilateral_image = cv2.bilateralFilter(image, d=9, sigmaColor=sigma, sigmaSpace=sigma)
        bilateral_image = Image.fromarray(bilateral_image)
        images.append(bilateral_image)

    elif 'diffusion' in args.filter_type:
        num_steps = 1000  # Number of diffusion steps

        # decide beta in each step
        betas = torch.linspace(-6,6,num_steps)
        betas = torch.sigmoid(betas) * (0.5e-2 - 1e-5) + 1e-5

        # decide alphas in each step
        alphas = 1 - betas
        alphas_prod = torch.cumprod(alphas, dim=0)
        alphas_prod_p = torch.cat([torch.tensor([1]).float(), alphas_prod[:-1]],0) # p for previous
        alphas_bar_sqrt = torch.sqrt(alphas_prod)
        one_minus_alphas_bar_log = torch.log(1 - alphas_prod)
        one_minus_alphas_bar_sqrt = torch.sqrt(1 - alphas_prod)
        
        if passing:
            image = np.array(image_file)
        else:
            try:
                image = cv2.imread(image_file.replace('jpg', 'png'))
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  
            except:
                image = Image.open(image_file).convert("RGB")
        image_tensor = image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
        
        def q_x(x_0, t):
            noise = torch.randn_like(x_0)
            alphas_t = alphas_bar_sqrt[t]
            alphas_1_m_t = one_minus_alphas_bar_sqrt[t]
            return (alphas_t*x_0 + alphas_1_m_t*noise)

        # breakpoint()
        noise_delta = int(sigma) # from 0-999
        noisy_image = image_tensor.clone()
        image_tensor_cd = q_x(noisy_image, sigma) 
        images.append(image_tensor_cd)
        # return image_tensor_cd

    return images


def add_filter(args, image_file, image_processor=None):
    images=[]
    passing = 1 if isinstance(image_file, PngImagePlugin.PngImageFile) or isinstance(image_file, JpegImagePlugin.JpegImageFile) or isinstance(image_file, Image.Image) else 0
    if 'gussian' in args.filter_type:
        if passing:
            image = np.array(image_file)
        else:
            image = cv2.imread(image_file.replace('jpg', 'png'))  
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  
        # image = cv2.GaussianBlur(image, (15, 15), 10)
        # image = Image.fromarray(image) 
        for i in range(2,10,2):
            gauss_image = cv2.GaussianBlur(image, (15, 15), i)
            gauss_image = Image.fromarray(gauss_image)
            images.append(gauss_image)
            
    elif 'median' in args.filter_type:
        if passing:
            image = np.array(image_file)
        else:
            image = cv2.imread( image_file.replace('jpg', 'png'))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        for i in range(3, 10, 2):
            median_image = cv2.medianBlur(image, i)
            median_image = Image.fromarray(median_image)
            images.append(median_image)
            
    elif 'sharpen' in args.filter_type:
        if passing:
            image = np.array(image_file)
        else:
            try:
                image = cv2.imread(image_file.replace('jpg', 'png'))
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  
            except:
                image = np.array(Image.open(image_file))
        for strength in [50, 150, 300, 500]:
            b = (1 - strength) / 8
            sharpening_kernel = np.array([[b, b, b],
                                        [b, strength, b],
                                        [b, b, b]])
            sharpened_image = cv2.filter2D(image, -1, sharpening_kernel)
            sharpened_image = Image.fromarray(sharpened_image, mode='RGB')
            images.append(sharpened_image)
        
    elif 'sobel' in args.filter_type:
        if passing:
            image = np.array(image_file)
        else:
            try:
                image = cv2.imread(image_file.replace('jpg', 'png'))
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  
            except:
                image = np.array(Image.open(image_file))
        # Sobel edge detection
        for i in range(3, 10, 2):  # Sobel 커널 크기를 3, 5로 변경
            sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=i)
            sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=i)
            sobel_image = cv2.magnitude(sobel_x, sobel_y)
            sobel_image = cv2.convertScaleAbs(sobel_image)
            sobel_image = Image.fromarray(sobel_image)
            images.append(sobel_image)
        
    elif 'bilateral' in args.filter_type:
        if passing:
            image = np.array(image_file)
        else:
            image = cv2.imread(image_file.replace('jpg', 'png'))  
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  
        for i in range(10, 50, 10):  # Sigma values for intensity and color space
            bilateral_image = cv2.bilateralFilter(image, d=9, sigmaColor=i, sigmaSpace=i)
            bilateral_image = Image.fromarray(bilateral_image)
            images.append(bilateral_image)
            
    elif 'diffusion' in args.filter_type:
        num_steps = 1000  # Number of diffusion steps

        # decide beta in each step
        betas = torch.linspace(-6,6,num_steps)
        betas = torch.sigmoid(betas) * (0.5e-2 - 1e-5) + 1e-5

        # decide alphas in each step
        alphas = 1 - betas
        alphas_prod = torch.cumprod(alphas, dim=0)
        alphas_prod_p = torch.cat([torch.tensor([1]).float(), alphas_prod[:-1]],0) # p for previous
        alphas_bar_sqrt = torch.sqrt(alphas_prod)
        one_minus_alphas_bar_log = torch.log(1 - alphas_prod)
        one_minus_alphas_bar_sqrt = torch.sqrt(1 - alphas_prod)

        # print(args.data_base_path, image_file)
        # breakpoint()
        if passing:
            image = image_file
        else:
            image = Image.open(image_file.replace('jpg', 'png'))
        image_tensor = image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
        
        def q_x(x_0, t):
            noise = torch.randn_like(x_0)
            alphas_t = alphas_bar_sqrt[t]
            alphas_1_m_t = one_minus_alphas_bar_sqrt[t]
            return (alphas_t*x_0 + alphas_1_m_t*noise)

        # noise_delta = int(sigma) # from 0-999
        for noise_step in [700, 800, 900, 999]:
            noisy_image = image_tensor.clone()
            image_tensor_cd = q_x(noisy_image, noise_step) 
            images.append(image_tensor_cd)

    return images