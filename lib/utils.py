import requests
import os
import tqdm
import numpy as np
import io
import base64
import cv2
import matplotlib.pyplot as plt
from functools import partial
from PIL import Image
import openai, logging
import tempfile
import torchvision.transforms.functional as F

from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

logger = logging.getLogger(__name__)
openai.api_key = os.environ["OPENAI_API_KEY"]

def crop_torch_im(im, x1, y1, x2, y2):
    '''crop a torch image (3, W, H)'''
    return im[:, int(y1):int(y2), int(x1):int(x2)]

def save_torch_image_tempfile(img, suffix='.png'):
    '''
    img: (3, W, H) torch tensor
    save torch image to a temporary file and return the path
    '''
    img = F.to_pil_image(img)
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
        img.save(f.name)
        filename = f.name
    return filename

def show_imgs(imgs):
    if not isinstance(imgs, list):
        imgs = [imgs]
    fig, axs = plt.subplots(ncols=len(imgs), squeeze=False)
    for i, img in enumerate(imgs):
        img = img.detach()
        img = F.to_pil_image(img)
        axs[0, i].imshow(np.asarray(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

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

def _x2img(x, prompt, negative_prompt,
           init_im=None, control_im=None,
           url='http://127.0.0.1:7860'):
    '''
    x is either txt or img
    requires automatic1111 api: https://github.com/Mikubill/sd-webui-controlnet/wiki/API#integrating-sdapiv12img
    running with the --api flag on url
    '''
    payload = {
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "sampler_name": "Euler",
        "steps": 10,
        "denoising_strength": 0.3, # default 0.75, lower means less noise
    }
    if init_im is not None:
        payload.update({"init_images": [im2string(init_im)]})
    if control_im is not None:
        payload.update({
            "alwayson_scripts": {
                "controlnet": {
                    "args": [
                        {
                            "input_image": im2string(control_im),
                            "module": "openpose",
                        "model": 'control_sd15_openpose [fef5e48e]'
                        }
                    ]
                }
            }
        })
    
    response = requests.post(url=f'{url}/sdapi/v1/{x}2img', json=payload)
    # Read results
    r = response.json()
    result = r['images'][0]
    return string2im(result)

def save_img(img, path):
    plt.imsave(path, img)

img2img = partial(_x2img, 'img')
txt2img = partial(_x2img, 'txt')

def create_retry_decorator(max_tries=3, min_seconds=4, max_seconds=10):
    # from langchain's _create_retry_decorator
    # Wait 2^x * 1 second between each retry starting with
    # 4 seconds, then up to 10 seconds, then 10 seconds afterwards
    return retry(
        reraise=True,
        stop=stop_after_attempt(max_tries),
        wait=wait_exponential(multiplier=1, min=min_seconds, max=max_seconds),
        retry=(
            retry_if_exception_type(openai.error.Timeout)
            | retry_if_exception_type(openai.error.APIError)
            | retry_if_exception_type(openai.error.APIConnectionError)
            | retry_if_exception_type(openai.error.RateLimitError)
            | retry_if_exception_type(openai.error.ServiceUnavailableError)
        ),
        before_sleep=before_sleep_log(logger, logging.WARNING),
    )

class ChatBot:
    '''open ai vannilla chatbot'''
    def __init__(self, system="", stop=None):
        self.system = system
        self.messages = []
        self.stop = stop # stop words
        if self.system:
            self.messages.append({"role": "system", "content": system})
    
    def __call__(self, message):
        self.messages.append({"role": "user", "content": message})
        result = self.execute()
        self.messages.append({"role": "assistant", "content": result})
        return result

    def save_chat(self, filename=""):
        '''save chat messages to json file, need to supply filename'''
        import json
        if filename != '':
            with open(filename, "w") as f:
                json.dump(self.messages, f)
                print(f'chat messages saved to {filename}')
        else:
            with tempfile.NamedTemporaryFile(mode='w+',
                                             delete=False) as f:
                json.dump(self.messages, f)
                print(f'Using tmpfile: {f.name}, as no filename is supplied')

    def load_chat(self, filename):
        '''load chat messages from json file'''
        import json
        self.message = json.load(open(filename, "r"))
            
    @create_retry_decorator(max_tries=3)
    def execute(self):
        completion = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=self.messages, stop=self.stop)
        # Uncomment this to print out token usage each time, e.g.
        # {"completion_tokens": 86, "prompt_tokens": 26, "total_tokens": 112}
        # print(completion.usage)
        return completion.choices[0].message.content
    


