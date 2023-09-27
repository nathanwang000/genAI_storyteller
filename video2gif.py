import click
import imageio
from lib.utils import video2frames

@click.command()
@click.option('-i', 'video_path', prompt=True, help='Input video file')
@click.option('-l', 'loop', default=0, help='Number of loops for gif (default inf)', type=int)
@click.option('-f', 'n_frames', default=0, help='Number of frames to convert (default all)', type=int)
def main(video_path, loop, n_frames):
    # Load the video file
    frames = video2frames(video_path, n_frames)

    video_name = video_path.split('/')[-1].split('.')[0]
    video_dir = '/'.join(video_path.split('/')[:-1])

    # Convert the video file to GIF
    imageio.mimsave(f"{video_dir}/{video_name}.gif", frames, 'GIF', loop=loop)
    # loop=0 means infinite loop, loop=1 means no loop, loop=2 means loop once, etc.

if __name__ == "__main__":
    main()







