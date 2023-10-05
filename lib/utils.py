import requests
import os
import tqdm
import numpy as np
import io
import base64
import cv2
import openai, logging
import tempfile
import torchvision.transforms.functional as F
import matplotlib.pyplot as plt
from PIL import Image, ImageEnhance

from duckduckgo_search import DDGS
from functools import partial
from PIL import Image
from PIL.ExifTags import TAGS

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
    '''crop a torch image (3, H, W)'''
    x1, x2 = int(max(min(x1, x2, im.shape[2]), 0)), int(min(max(x1, x2, 0), im.shape[2]))
    y1, y2 = int(max(min(y1, y2, im.shape[1]), 0)), int(min(max(y1, y2, 0), im.shape[1]))
    return im[:, y1:y2, x1:x2]

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
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    # Release the video capture object
    video.release()

    if n_frames <= 0: # use full frames
        n_frames = len(frames)

    n_skip = max(len(frames) // n_frames, 1)
    return frames[::n_skip][:n_frames]

def _x2img(x, prompt, negative_prompt,
           init_im=None, control_im=None,
           denoising_strength=0.3,
           steps=10,
           width=848, height=480,
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
        "steps": steps,
        'width': width,
        'height': height,
        "denoising_strength": denoising_strength, # default 0.75, lower means less noise
    }
    if init_im is not None:
        payload.update({"init_images": [im2string(init_im[:,:,::-1])]}) # convert to bgr b/c it will invert back
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

img2img = partial(_x2img, 'img')
txt2img = partial(_x2img, 'txt')
def save_img(img, path):
    plt.imsave(path, img)

def save_torch_img(img, path):
    F.to_pil_image(img).save(path)

def get_capture_time(image_path):
    """
    Extracts the capture time from the metadata of an image.
    
    Args:
        image_path (str): Path to the image file.

    Returns:
        str: Capture time in a human-readable format (e.g., "2023-06-02 15:30:45").
        "": If capture time is not found or an error occurs.
    """
    try:
        image = Image.open(image_path)
        exif_data = image._getexif()

        # Iterate over the EXIF data and search for the capture time tag
        for tag_id, value in exif_data.items():
            tag_name = TAGS.get(tag_id, tag_id)
            if tag_name == 'DateTimeOriginal':
                return value

        return ''  # Capture time tag not found
    except (AttributeError, KeyError, IndexError):
        return ''  # Error occurred during extraction

def download_image(url, save_path):
    """
    Downloads an image from a given URL and saves it to a specified path.

    Args:
        url (str): URL of the image.
        save_path (str): Path to save the downloaded image.
    
    Returns:
        bool: True if the download was successful, False otherwise.
    """
    try:
        response = requests.get(url)
        response.raise_for_status()
        
        with open(save_path, 'wb') as file:
            file.write(response.content)
            
        return True  # Download successful
    except requests.exceptions.RequestException:
        return False  # Error occurred during download

def download_ddgs_image_search(search_term,
                               max_images=1000,
                               output_dir='output/ddgs_images',
                               **kwargs):
    os.system(f'mkdir -p {output_dir}/{"_".join(search_term.split())}')
    i = 0
    with DDGS() as ddgs:
        keywords = search_term,
        ddgs_images_gen = ddgs.images(
            keywords,
            **kwargs
        )
        progress_bar = tqdm.tqdm(total=max_images)
        for r in ddgs_images_gen:
            progress_bar.set_description(f"Downloading: {i}/{max_images}")
            progress_bar.update(1)
            i += 1
            if i > max_images:
                break
            url = r['image']
            postfix = url.split('.')[-1]
            if postfix not in ['jpg', 'png', 'jpeg']:
                print('unknown postfix', postfix)
                continue
            download_image(url, f'{output_dir}/{"_".join(search_term.split())}/{i}.{postfix}')

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
    
def image_enhance(fn, contrast_factor=1.5, color_factor=1.5):
    # Open the image
    if type(fn) is str:
        img = Image.open(fn)
    else:
        img = Image.fromarray(fn)
    
    # Enhance the color (saturation)
    color_enhancer = ImageEnhance.Color(img)
    img_colored = color_enhancer.enhance(color_factor)
    
    # Enhance the contrast
    contrast_enhancer = ImageEnhance.Contrast(img_colored)
    img_contrasted = contrast_enhancer.enhance(contrast_factor)
        
    # return the enhanced image as a numpy array only preserving the RGB channels
    return np.array(img_contrasted)[:,:,:3]
