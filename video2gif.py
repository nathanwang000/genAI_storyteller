import click
import imageio
from lib.utils import video2frames

@click.command()
@click.option('-i', 'video_path', prompt=True, help='Input video file')
def main(video_path):
    # Load the video file
    frames = video2frames(video_path)

    video_name = video_path.split('/')[-1].split('.')[0]
    video_dir = '/'.join(video_path.split('/')[:-1])

    # Convert the video file to GIF
    imageio.mimsave(f"{video_dir}/{video_name}.gif", frames)

if __name__ == "__main__":
    main()
