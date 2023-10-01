import cv2
import os
import click
from lib.utils import video2frames

@click.command()
@click.option('-i', 'video_path', prompt=True, help='Input video file')
@click.option('-f', 'n_frames', default=0, help='Number of frames to convert (default all)', type=int)
@click.option('--fps', default=24, help='output video file')
def main(video_path, n_frames, fps):
    # Load the video file
    frames = video2frames(video_path, n_frames)
    # concat with the reverse frames
    frames = frames + frames[::-1]

    video_name = video_path.split('/')[-1].split('.')[0]
    video_dir = '/'.join(video_path.split('/')[:-1])
    
    # Create a video writer object
    output_video_path = os.path.join(video_dir,
                                     f'{video_name}_inverse_loop.mp4')
    height, width, _ = frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_video_path,
                                   fourcc, fps,
                                   (width, height))

    # Write each image to the video writer
    for frame in frames:
        # Write the frame to the video writer
        # convert to cv2 format
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        video_writer.write(frame)

    # Release the video writer
    video_writer.release()

    print(f"Video created successfully: {output_video_path}")



if __name__ == "__main__":
    main()
