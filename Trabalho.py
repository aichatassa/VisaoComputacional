import cv2
import numpy as np
import re
img_folder = 'chromas/'
output_folder = 'resultados/'
file_bckgd = 'background.bmp'
background = cv2.imread(img_folder+file_bckgd)
array_imgs = ["corvo.bmp", "corvos.bmp", "formas.bmp", "rainha.bmp"]

def chroma_key(array_imgs, img_folder, output_folder, file_bckgd, background):
  for foreground in array_imgs:
    img = cv2.imread(img_folder+foreground)

    height, width, channels = img.shape

    resize_bckgd = cv2.resize(background, (width, height))

    blur_bckgd = cv2.GaussianBlur(resize_bckgd, (5, 5), 0)
    
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    bg_hsv = cv2.cvtColor(blur_bckgd, cv2.COLOR_BGR2HSV)

    lower_green = np.array([50, 100, 100])
    upper_green = np.array([70, 255, 255])

    green_mask = cv2.inRange(hsv, lower_green, upper_green)

    _, mask = cv2.threshold(green_mask, 0, 255, cv2.THRESH_BINARY)
    cv2.imwrite(foreground.replace(".bmp", "/")+foreground.replace(".bmp", "_binarizada.bmp"), mask)
    
    #Suavização com BilateralFilter
    #Suavização com GaussianBlur não mostrou diferença com relação ao Bilateral Filter
    sigmaColor = 75
    sigmaSpace = 75
    blur = cv2.bilateralFilter(mask, 20, sigmaColor, sigmaSpace)
    cv2.imwrite(output_folder+foreground.replace(".bmp", "_bilateral.bmp"), blur)
    

    masked_bg = cv2.bitwise_and(bg_hsv, bg_hsv, mask=mask)

    mask_inv = cv2.bitwise_not(mask)

    masked_img = cv2.bitwise_and(hsv, hsv, mask=mask_inv)

    result_hsv = cv2.add(masked_bg, masked_img)

    result_bgr = cv2.cvtColor(result_hsv, cv2.COLOR_HSV2BGR)
    cv2.imwrite(output_folder+foreground, result_bgr)

chroma_key(array_imgs, img_folder, output_folder, file_bckgd, background)
print("Resultados salvos na pasta referenciada no arquivo output_folder")