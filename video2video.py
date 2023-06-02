'''based on automatic1111 api: https://github.com/Mikubill/sd-webui-controlnet/wiki/API#integrating-sdapiv12img'''
import requests
import os
import click
import tqdm
import numpy as np
import io
import base64
import cv2
import matplotlib.pyplot as plt
from PIL import Image

def string2im(s):
    '''from string to image'''
    image = Image.open(io.BytesIO(base64.b64decode(s.split(",", 1)[0])))
    return np.array(image)

def im2string(img):
    '''from numpy image to string'''
    retval, bytes = cv2.imencode('.png', img)
    encoded_image = base64.b64encode(bytes).decode('utf-8')
    return encoded_image

def remove_ext(fname):
    '''remove file extension'''
    return os.path.basename(os.path.splitext(fname)[0])

import cv2

def video2frames(video_path, n_frames=0):
    '''convert video to frames, if n_frames is 0, convert all frames,
    otherwise convert n_frames frames by evenly sampling'''
    # Open the video file
    video = cv2.VideoCapture(video_path)

    frames = []

    while True:
        # Read the next frame from the video
        ret, frame = video.read()

        # Break the loop if no more frames are available
        if not ret:
            break

        # Append the frame to the list
        frames.append(frame)

    # Release the video capture object
    video.release()

    if n_frames <= 0: # use full frames
        n_frames = len(frames)

    n_skip = max(len(frames) // n_frames, 1)
    return frames[::n_skip][:n_frames]

def img2img(prompt, negative_prompt, init_im, control_im):
    init_im_str = im2string(init_im)
    control_im_str = im2string(control_im)
    payload = {
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "init_images": [init_im_str],
        "sampler_name": "Euler",
        "steps": 10,
        "alwayson_scripts": {
            "controlnet": {
                "args": [
                    {
                        "input_image": control_im_str,
                        "module": "openpose",
                        "model": 'control_sd15_openpose [fef5e48e]'
                    }
                ]
            }
        }
    }
    url = "http://127.0.0.1:7860"
    response = requests.post(url=f'{url}/sdapi/v1/txt2img', json=payload)
    # Read results
    r = response.json()
    result = r['images'][0]
    return string2im(result)

def save_img(img, path):
    plt.imsave(path, img)

@click.command()
@click.option('-p', 'prompt', prompt=True, default='tiger', help='prompt for stable diffusion')
@click.option('-n', 'negative_prompt', prompt=True, default='worst-quality',
              help='negative prompt for stable diffusion')
@click.option('-v', 'video_path', prompt=True, default='./example.mp4',
              help='video_path for stable diffusion')
@click.option('-f', 'n_frames', prompt=True, default=0, type=int, help='number of frames')
@click.option('-o', 'output_dir', default='./output',
              help='output directory for the resulting video')
def main(prompt, negative_prompt, video_path, n_frames, output_dir):
    os.system('mkdir -p {}'.format(output_dir))
    frames = video2frames(video_path, n_frames)
    init_im = frames[0]
    for i, control_im in enumerate(tqdm.tqdm(frames, desc='Generating video')):
        # using init_im as the initial image and frame as control image
        init_im = img2img(prompt, negative_prompt, init_im, control_im)
        save_img(init_im, os.path.join(output_dir, f'{remove_ext(video_path)}_{i}.png'))

if __name__ == '__main__':
    main()
