import click
import imageio
from lib.utils import video2frames

@click.command()
@click.option('-i', 'video_path', prompt=True, help='Input video file')
@click.option('-f', 'n_frames', default=0, help='Number of frames to convert (default all)', type=int)
def main(video_path, n_frames):
    # Load the video file
    frames = video2frames(video_path, n_frames)

    video_name = video_path.split('/')[-1].split('.')[0]
    video_dir = '/'.join(video_path.split('/')[:-1])

    # Save the frames
    for i, frame in enumerate(frames):
        imageio.imwrite(f'{video_dir}/{i}.png', frame)


if __name__ == "__main__":
    main()
