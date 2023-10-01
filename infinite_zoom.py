import os
import click
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

def get_interesting_patch(img, nh, nw, center=True,
                          reference_img=None, n_windows=5):
    '''
    nh, nw: margin size on height and width
    n_windows: number of windows**2 to try
    '''
    h, w, _ = img.shape

    if reference_img is not None:
        # compute gram's matrix
        g = gram_matrix(reference_img)
        
    if center and reference_img is None:
        return img[nh:-nh,nw:-nw,:]

    max_var = -float('inf')
    max_i, max_j = 0, 0
    for _i in range(0, min(nh*n_windows, h-2*nh), nh):
        for _j in range(0, min(nw*n_windows, w-2*nw), nw):
            # todo: optimize this and add path regularization
            # todo: try to use a color dictionary instead of gram's matrix
            # get the most interesting patch
            patch = img[_i:_i+h-2*nh,_j:_j+w-2*nw,:]
            if reference_img is None:
                img_var = np.var(patch)
            else:
                img_var = -((g - gram_matrix(patch))**2).mean() # want to minimize l2
    
            if img_var > max_var:
                max_var = img_var
                max_i, max_j = _i, _j
    return img[max_i:max_i+h-2*nh,max_j:max_j+w-2*nw,:]

@click.command()
@click.option('--output_dir', '-o', default='output/test', help='output directory')
@click.option('--prompt', '-p', default='galaxy, stars, cinematic, colorful, mgical girl, hd',
              type=str,
              help='prompt for generating the image')
@click.option('--negative_prompt', '-np', default='ugly, disfigured, watermark',
              type=str,
              help='negative prompt')
@click.option('--margin', '-m', default=0.02, type=float, help='margin size for cropping/zooming')
@click.option('--n_steps', '-n', default=100, type=int, help='number of images to generate')
@click.option('--use_reference', flag_value=True, help='use reference image to guide where to zoom, o/w zoom to the center')
@click.option('--min_ds', default=0.4, type=float, help='minimum denoising strength, higher means ignoring image more')
@click.option('--max_ds', default=0.8, type=float, help='maximum denoising strength, higher means ignoring image more')
def zoom(output_dir, prompt, negative_prompt, margin, n_steps, use_reference, min_ds, max_ds):
    os.makedirs(output_dir, exist_ok=True)    
    # prompt = 'bird singing on a tree, hd, cinematic lighting'
    # prompt = 'cyberpunk city, tokyo'

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
                if ratio > 1.2:
                    denoising_strength = max(0.8 * denoising_strength, min_ds)
                img = img2

        save_img(img, os.path.join(output_dir, f'{i}.png'))

if __name__ == '__main__':
    zoom()
    

