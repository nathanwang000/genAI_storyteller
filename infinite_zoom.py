# todo: make it more generalizable
# todo: add interesting shot chasing shots: match on gram's matrix
import os
import tqdm
import numpy as np
from lib.utils import txt2img, img2img, save_img
from skimage.transform import resize

def gram_matrix(im):
    '''
    assumes the last dimension is the channel
    '''
    c = im.reshape(-1, im.shape[-1])
    return c.T@c

def get_interesting_patch(img, nh, nw, center=True, reference_img=None):
    '''
    nh, nw: margin size on height and width
    '''
    h, w, _ = img.shape

    if reference_img is not None:
        # compute gram's matrix # could do a color dictionary matching instead
        g = gram_matrix(reference_img)
        
    if center and reference_img is None:
        return img[nh:-nh,nw:-nw,:]

    max_var = -float('inf')
    max_i, max_j = 0, 0
    for _i in range(0, min(nh*5, h-2*nh), nh):
        for _j in range(0, min(nw*5, w-2*nw), nw):
            # todo: optimize this and add path regularization
            # get the most interesting patch
            # TODO: calculate added variance (new point distance to its closest point)
            patch = img[_i:_i+h-2*nh,_j:_j+w-2*nw,:]
            if reference_img is None:
                img_var = np.var(patch)
            else:
                img_var = -((g - gram_matrix(patch))**2).mean() # want to minimize l2
    
            if img_var > max_var:
                max_var = img_var
                max_i, max_j = _i, _j
    return img[max_i:max_i+h-2*nh,max_j:max_j+w-2*nw,:]
    
def zoom():
    output_dir = 'output/test'
    os.makedirs(output_dir, exist_ok=True)    
    # prompt = 'bird singing on a tree, hd, cinematic lighting'
    # prompt = 'cyberpunk city, tokyo'
    prompt = 'galaxy, stars, cinematic, colorful, mgical girl, hd'
    negative_prompt = 'ugly, disfigured, watermark'
    min_ds, max_ds = 0.4, 0.8
    margin = 0.02
    n_steps = 300
    use_reference = False

    denoising_strength = min_ds
    img_diffs = []
    # trange with description changes the tqdm bar description
    progress_bar = tqdm.tqdm(range(n_steps), desc='')
    for i in progress_bar:
        if i == 0:
            init_img = txt2img(prompt, negative_prompt)
            img = init_img
        else:
            # zoom in img
            h, w, _ = img.shape
            nh, nw = max(int(margin * h), 1), max(int(margin * w), 1)
            desired_size = h, w

            # size down the next frame
            img = get_interesting_patch(img, nh, nw, reference_img=init_img if use_reference else None)
            
            # resize img back; resize is float so we need to map back to uint8
            img = (resize(img, desired_size, anti_aliasing=True) * 255).astype(np.uint8)

            if i % 2 == 0:
                img2 = img2img(prompt, negative_prompt, init_im=img,
                               denoising_strength=denoising_strength)

                # dynamically adjust denoising_strength
                img_diffs.append((abs(img-img2)).mean())
                progress_bar.set_description(f'img diff: {img_diffs[-1]:.2f}, denoise: {denoising_strength:.2f}')
                ratio = img_diffs[-1] / max(img_diffs)
                if ratio < 0.8:
                    denoising_strength = min(1.2 * denoising_strength, max_ds)
                    # # add random white spots to img2; maybe later add to latent space
                    # n = int(30 * (1 - ratio))
                    # for _ in range(n):
                    #     x, y = np.random.randint(0, h), np.random.randint(0, w)
                    #     img2[x:x+nh,y:y+nw,:] = 255
                if ratio > 1.2:
                    denoising_strength = max(0.8 * denoising_strength, min_ds)
                img = img2

        save_img(img, os.path.join(output_dir, f'{i}.png'))

if __name__ == '__main__':
    zoom()
    

