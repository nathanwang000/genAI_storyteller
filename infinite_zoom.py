# todo: make it more generalizable
import os
import numpy as np
from lib.utils import txt2img, img2img, save_img
from skimage.transform import resize

def zoom():
    output_dir = 'output/test'
    os.makedirs(output_dir, exist_ok=True)
    for i in range(30):
        if i == 0:
            img = txt2img('universe, stars, magic girl', 'ugly, disfigured')
        else:
            # zoom in img
            n = 100
            desired_size = img.shape[0], img.shape[1]
            img = img[n:-n,n:-n,:]
            # resize img back; resize is float so we need to map back to uint8
            img = (resize(img, desired_size, anti_aliasing=True) * 255).astype(np.uint8)
            save_img(img, os.path.join(output_dir, f'{i}_resize.png'))
            img = img2img('universe, stars', 'ugly, disfigured', init_im=img,
                          denoising_strength=0.75) # 0.5
    
        save_img(img, os.path.join(output_dir, f'{i}.png'))

if __name__ == '__main__':
    zoom()
    

