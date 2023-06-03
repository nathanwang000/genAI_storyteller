import click, os, tqdm
from lib.utils import video2frames, img2img, txt2img, save_img, remove_ext

@click.command()
@click.option('-p', 'prompt', prompt=True, default='tiger', help='prompt for stable diffusion')
@click.option('-n', 'negative_prompt', prompt=True, default='worst-quality, watermark',
              help='negative prompt for stable diffusion')
@click.option('-v', 'input_video_path', prompt=True, default='./example.mp4',
              help='video_path for stable diffusion')
@click.option('-f', 'n_frames', prompt=True, default=0, type=int, help='number of frames')
@click.option('-o', 'output_dir', prompt=True, default='./output',
              help='output directory for the resulting video')
def main(prompt, negative_prompt, input_video_path, n_frames, output_dir):
    os.system('mkdir -p {}'.format(output_dir))
    frames = video2frames(input_video_path, n_frames)
    init_im = frames[0]
    for i, control_im in enumerate(tqdm.tqdm(frames, desc='Generating video')):
        # using init_im as the initial image and frame as control image
        if i == 0:
            init_im = txt2img(prompt, negative_prompt, init_im, control_im)
        else:
            init_im = img2img(prompt, negative_prompt, init_im, control_im)
        save_img(init_im, os.path.join(output_dir, f'{remove_ext(input_video_path)}_{i}.png'))

if __name__ == '__main__':
    main()
