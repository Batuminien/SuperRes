import cv2
import numpy as np
import matplotlib.pyplot as plt
from src.utils import imread_normalized, rgb2yiq, yiq2rgb, imsave_normalized
from src.sr_solver import SRSolver

def main():
    #settings
    input_path = "input/Set5/Set5/butterfly.png"
    target_scale = 2.0
    
    print(f"Operation is starting: {input_path}")
    print(f"Target Scale: x{target_scale}")
    
    #load the image and turns it into YIQ
    print(f"Image is loading...")
    hr_img = imread_normalized(input_path)
    
    h, w, c = hr_img.shape
    downscale_factor = 2
    new_h = h // downscale_factor
    new_w = w // downscale_factor
    
    img_input = cv2.resize(hr_img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
    
    print(f"Original Size: {hr_img.shape}")
    print(f"Input for SR (Low-Res): {img_input.shape}")
    
    img_yiq = rgb2yiq(img_input)
    
    #just take Y (luminance) channel
    y_channel = img_yiq[:, :, 0]
    #keep the color channel (I,Q), we upscale them simply after.
    i_channel = img_yiq[:, :, 1]
    q_channel = img_yiq[:, :, 2]
    
    #super resolution
    solver = SRSolver(y_channel)
    sr_y = solver.upscale(target_scale=target_scale)
    
    #upscale the color channel (bicubic)
    h_sr, w_sr = sr_y.shape
    sr_i = cv2.resize(i_channel, (w_sr, h_sr), interpolation=cv2.INTER_CUBIC)
    sr_q = cv2.resize(q_channel, (w_sr, h_sr), interpolation=cv2.INTER_CUBIC)
    
    #concatanate the channels and converts it into RGB
    sr_yiq = np.dstack((sr_y, sr_i, sr_q))
    sr_rgb = yiq2rgb(sr_yiq)
    
    #comparison (standart bicubic)
    #to see the difference we upscale the original image simply
    bicubic_rgb = cv2.resize(img_input, (w_sr, h_sr), interpolation=cv2.INTER_CUBIC)
    
    sr_rgb = np.clip(sr_rgb, 0.0, 1.0)          
    bicubic_rgb = np.clip(bicubic_rgb, 0.0, 1.0)
    
    #save the result and show
    imsave_normalized("output/result_bicubic.png", bicubic_rgb)
    imsave_normalized("output/result_ours.png", sr_rgb)
    
    print("\nOperation is done.")
    
    plt.figure(figsize=(12, 6))
    
    #bicubic result
    plt.subplot(1, 2, 1)
    plt.imshow(bicubic_rgb)
    plt.title("Standart Bicubic")
    plt.axis('off')

    #our result
    plt.subplot(1, 2, 2)
    plt.imshow(sr_rgb)
    plt.title("Our SR Method")
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()