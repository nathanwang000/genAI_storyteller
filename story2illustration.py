import click, json, os
from lib.utils import img2img, txt2img, save_img
from lib.utils import ChatBot

def parse_story_response(response:str):
    '''response is a json string: list of story prompt'''
    res = json.loads(response)
    for story in res:
        if 'story' not in story or 'prompt' not in story:
            raise ValueError('Invalid response')
    return res

@click.command()
@click.option('--story_prompt', prompt=True,
              default='Once upon a time, there was a little tiger named lua',
              help='llm prompt')
@click.option('--output_dir', prompt=True, default='output',
              help='output directory')
def main(story_prompt, output_dir):
    system_prompt = '''
    User will input a story prompt, your job is to output a json file that
    a) complete the story and
    b) provide prompt to the stable diffusion model to illustrate the story.

    Break your story in to chunks to create multiple illustrations.
    The output json should be a list containing the fields "story" and "prompt" where "story" is your completion and "prompt" is fed into stable diffusion.

    Here is an example output
    ```[{"story": <story chunk 1>, "prompt": <story chunk 1's prompt to stable diffusion>}, {"story": <story chunk 2>, "prompt": <story chunk 2's prompt to stable diffusion>}]```
    '''
    os.system(f'mkdir -p {output_dir}')
    chatbot = ChatBot(system_prompt)
    response = chatbot(story_prompt)
    try:
        r_dict = parse_story_response(response)
    except Exception as e:
        print(e)
        print('response not formatted correctly, exiting...')
        return

    story_fn = os.path.join(output_dir, 'story.json')
    json.dump(r_dict, open(story_fn, 'w'), indent=4)
    print(f'done generating the story, saved in {story_fn}')
    for i, story in enumerate(r_dict):
        story_text = story['story']
        prompt = story['prompt']
        img = txt2img(prompt, "worst-quality, watermark") # todo: customize later
        save_img(img,
                 os.path.join(output_dir, f'{i}.png'))

if __name__ == '__main__':
    main()
