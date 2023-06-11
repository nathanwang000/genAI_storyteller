import torch
import tqdm
import click
import glob
import cv2
import os
import torchvision
import face_alignment
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms.functional as F

import torchvision.transforms as T
from torchvision.utils import draw_keypoints
from torchvision.utils import draw_bounding_boxes
from torchvision.io import read_image
from torchvision.io import ImageReadMode
from deepface import DeepFace

from lib.utils import show_imgs, crop_torch_im
from lib.utils import save_torch_image_tempfile

def apply_theta(kpts, theta):
    return np.hstack([kpts, np.ones((len(kpts), 1))]) @ theta.T

def eval_theta(src_kpts, dst_kpts, theta):
    return ((apply_theta(src_kpts, theta) - dst_kpts)**2).sum()
        
def align_kpts(src_kpts, dst_kpts):
    '''
    linear transformation to align src_kpts to dst_kpts
    src_kpts: (N, 2) ndarray
    dst_kpts: (N, 2) ndarray
    return theta: (2, 3) trnasformation matrix
    that is, dst_kpts = theta * src_kpts
    '''
    # src_kpts: (N, 2)
    # dst_kpts: (N, 2)
    # theta: (2, 3)
    assert src_kpts.shape == dst_kpts.shape
    theta = cv2.estimateAffinePartial2D(src_kpts, dst_kpts)[0]
    return theta

def align_kpts2(src_kpts, dst_kpts):
    '''
    linear transformation to align src_kpts to dst_kpts
    src_kpts: (N, 2) ndarray
    dst_kpts: (N, 2) ndarray
    min |dst_kpts - (src_kpts,1) theta|^2
    return theta.T: (3,2)
    '''
    assert src_kpts.shape == dst_kpts.shape
    N = len(src_kpts)
    theta, _, _, _ = np.linalg.lstsq(
        np.hstack([src_kpts, np.ones((N,1))]),
        dst_kpts, rcond=None)
    return theta.T
    
def apply_affine_transformation(image, transformation_matrix):
    """
    Applies an affine transformation to an image.
    
    Args:
        image (ndarray): Input image as a NumPy array. (H, W, C)
        transformation_matrix (ndarray): 2x3 transformation matrix.

    Returns:
        ndarray: Transformed image as a NumPy array.
    """
    height, width = image.shape[:2]
    transformed_image = cv2.warpAffine(image, transformation_matrix, (width, height))
    return transformed_image

def align_face(img, src_kpts, dst_kpts, ransac=False):
    '''
    img: (3, H, W) tensor
    src_kpts: (N, 2)
    dst_kpts: (N, 2)
    output: (3, H, W) tensor
    '''
    if ransac:
        f_align = align_kpts
    else:
        f_align = align_kpts2
        
    theta = f_align(src_kpts, dst_kpts)
    im = img.permute((1, 2, 0)).numpy()
    np_out = apply_affine_transformation(im,
                                         theta)
    return torch.from_numpy(np_out).permute((2, 0, 1))

def save_faces(face_images, output_dir):
    for i, face in enumerate(face_images):
        pil_face = T.ToPILImage()(face)
        pil_face.save(os.path.join(output_dir, f'{i}.png'))

def face_distance(img1, img2):
    '''
    img1: (3, H, W) tensor
    img2: (3, H, W) tensor
    '''
    return DeepFace.verify(
        img1_path=save_torch_image_tempfile(img1),
        img2_path=save_torch_image_tempfile(img2),
        enforce_detection=False)['distance']

def get_key_points(im, device='mps'):
    fa = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D,
                                      device=device)
    return fa.get_landmarks(im, return_bboxes=True)
        
def align_faces_dir(drive_image_path, image_dir, output_dir, ransac=False,
                    device='mps', kpt_radius=0,
                    width=450, height=450, crop_pad=None):
    print('currently just aligning png and jpg files...')
    imps = [drive_image_path] + glob.glob(f'{image_dir}/*.png') + glob.glob(f'{image_dir}/*.png')
    imgs = []
    lmks = []
    bboxes = []
    for i, imp in enumerate(tqdm.tqdm(imps,
                                      desc='extracting face keypoints in images')):
        im = torchvision.transforms.Resize((width, height))(
            read_image(imp, ImageReadMode.RGB))
        lmk, lmk_socres, bbox = get_key_points(
            im if im.shape[2] == 3 else im.permute((1,2,0)),
            device=device)

        if i == 0: # drive image
            if not len(bbox):
                raise ValueError(f'no face detected in drive image {imp}')
            elif len(bbox) > 1:
                raise ValueError(f'multiple faces detected in drive image {imp}')
        else: # other images
            if len(bbox):
                # choose the face with the lowest distance with drive image
                distances = [face_distance(
                    crop_torch_im(im, *bbox[i][:4]),
                    crop_torch_im(imgs[0], *bboxes[0][0][:4]))\
                             for i in range(len(bbox))]
                if min(distances) > 0.4:
                    print(f'{imp} likely not have the driver in {imps[0]}')
                bbox = [bbox[np.argmin(distances)]]
            if not len(bbox):
                print(f'no driving face detected in image {imp}')
                continue

        lmks.append(lmk)
        imgs.append(
            draw_keypoints(
                im,
                torch.from_numpy(np.array(lmk)), colors="blue", radius=kpt_radius)
        )
        bboxes.append(bbox)
                    
    aligned_faces = []
    for i, img in enumerate(tqdm.tqdm(imgs, desc='aligning faces')):
        if i == 0:
            aligned_face = img
        else:
            # use the first one as it is verified to be the drive image person or no need to match driver
            aligned_face = align_face(img, lmks[i][0], lmks[0][0])
        if crop_pad is not None:
            a,b,c,d = list(map(int, bboxes[0][0][:4])) # drive image
            pad = crop_pad
            a, b, c, d = max(a-pad, 0), max(b-pad, 0), min(c+pad, width), min(d+pad, height)
            aligned_face = aligned_face[:, b:d, a:c]
        aligned_faces.append(aligned_face)
        
    os.system(f'mkdir -p {output_dir}')
    save_faces(aligned_faces, output_dir)

@click.command()
@click.option('--drive-image-path', '-d', required=True,
              help='path to the image to align with')
@click.option('--image_dir', '-i', required=True, help='image dir')
@click.option('--output_dir', '-o', prompt=True, help='output dir')
@click.option('--ransac', default=False, help='use ransac')
@click.option('--device', default='mps', help='device to run face alignment')
@click.option('--kpt_radius', default=0, help='radius to visualize keypoints')
@click.option('--width', default=450, help='width of aligned faces')
@click.option('--height', default=450, help='height of aligned faces')
@click.option('--crop_pad', default=None, type=int, help='pad to crop faces')
def main(drive_image_path, image_dir,
         output_dir, ransac,
         device, kpt_radius,
         width, height, crop_pad):
    align_faces_dir(drive_image_path, image_dir,
                    output_dir, ransac,
                    device, kpt_radius,
                    width, height, crop_pad)
    
if __name__ == '__main__':
    main()






