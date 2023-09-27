import cv2
import os
import click

@click.command()
@click.option('--inverse_video', '-i',
              help='whether to invert images to loop backwards', is_flag=True, default=False)
@click.option('--images_folder', prompt=True, default='./output', help='image directory')
@click.option('--output_video_path', prompt=True, default='output.mp4', help='output video file')
@click.option('--fps', prompt=True, default=24, help='output video file')
@click.option('--image_postfix', prompt=True, default='png', help='image postfix')
def create_video(inverse_video, images_folder, output_video_path, fps, image_postfix):
    # Get the list of image filenames in the folder
    image_filenames = sorted([fn for fn in os.listdir(images_folder) if fn.endswith(f'.{image_postfix}')],
                             key=lambda x: int(x.split('_')[-1].split('.')[0]))
    print(image_filenames)
    if inverse_video:
        print('loop in inverted video order')
        image_filenames = image_filenames + image_filenames[::-1]

    # Get the dimensions of the first image
    first_image_path = os.path.join(images_folder, image_filenames[0])
    first_image = cv2.imread(first_image_path)
    height, width, _ = first_image.shape

    # Create a video writer object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    # Write each image to the video writer
    for image_filename in image_filenames:
        image_path = os.path.join(images_folder, image_filename)
        image = cv2.imread(image_path)
        video_writer.write(image)

    # Release the video writer
    video_writer.release()

    print(f"Video created successfully: {output_video_path}")

if __name__ == "__main__":
    create_video()

