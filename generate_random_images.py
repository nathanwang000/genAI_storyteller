import os
import click
import tqdm
import numpy as np
from lib.utils import txt2img, img2img, save_img
from skimage.transform import resize

@click.command()
@click.option('--output_dir', '-o', default='output/random', help='output directory')
@click.option('--negative_prompt', '-np', default='ugly, disfigured, watermark',
              type=str,
              help='negative prompt')
@click.option('--n_steps', '-n', default=10, type=int, help='number of images to generate')
def main(output_dir, negative_prompt, n_steps):
    os.makedirs(output_dir, exist_ok=True)    

    professions = ["Astronaut", "Marine Biologist", "Forensic Anthropologist", "Cinematographer", "Beekeeper (Apiarist)", "Urban Planner", "Nuclear Physicist", "Ethnomusicologist", "Roofer", "Dietitian"]

    places = ["Ocean", "Desert", "Backyard", "Mountain Peak", "Cave", "Forest", "Island", "Riverbank", "City Square", "Prairie", "gym"]

    actions = ["Whispering to a willow tree", "Dancing atop a dew-laden morning grass", "Juggling fireflies on a moonless night", "Sketching dreams onto foggy windows", "Composing a symphony for crickets", "Weaving tales with a spider's silk", "Balancing raindrops on the tip of a feather", "Painting the horizon using only the hues of emotions", "Catching stardust in a jar during a meteor shower", "Planting ideas in a garden of curiosity"]

    progress_bar = tqdm.tqdm(range(n_steps), desc='')
    for i in progress_bar:
        prompt = f"A {np.random.choice(professions).lower()} is {np.random.choice(actions).lower()}, near {np.random.choice(places).lower()}."
        img = txt2img(prompt, negative_prompt, steps=30)
        save_img(img, os.path.join(output_dir, f"{'_'.join(prompt.split())}.png"))

if __name__ == '__main__':
    main()
    

