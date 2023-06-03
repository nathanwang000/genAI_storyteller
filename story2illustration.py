import click, json, os, tqdm
from lib.utils import img2img, txt2img, save_img
from lib.utils import ChatBot


def parse_story_response(response:str):
    '''response is a json string: list of story prompt'''
    res = json.loads(response)
    for story in res:
        if 'text' not in story or 'illustration' not in story:
            raise ValueError('Invalid response')
    return res

@click.command()
@click.option('--story_prompt', prompt=True,
              default='Once upon a time, there was a little tiger named lua',
              help='llm prompt')
@click.option('--output_dir', prompt=True, default='output',
              help='output directory')
@click.option('--system_prompt_path', prompt=True, default='prompts/story_system_prompt0.md',
              help='system prompt path')
def main(story_prompt, output_dir, system_prompt_path):
    system_prompt = open(system_prompt_path).read()
    os.system(f'mkdir -p {output_dir}')
    chatbot = ChatBot(system_prompt)
    print('Querying the llm for story completion...')
    story_prompt = f'Story to complete: {story_prompt}\n\nRemember to respond in json with the correct format!'
    response = chatbot(story_prompt)
    try:
        r_dict = parse_story_response(response)
    except Exception as e:
        print(e)
        print(response)
        print('response not formatted correctly, exiting...')
        return

    story_fn = os.path.join(output_dir, 'story.json')
    json.dump(json.loads(response), open(story_fn, 'w'), indent=4)
    print(f'Done generating the story, saved in {story_fn}, starting to generate illustrations...')
    for i, story in enumerate(tqdm.tqdm(r_dict)):
        img = txt2img(story['illustration'], "worst-quality, watermark")
        save_img(img,
                 os.path.join(output_dir, f'{i}.png'))

if __name__ == '__main__':
    main()
