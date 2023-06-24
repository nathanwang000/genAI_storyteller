import click, json, os, tqdm
from lib.utils import img2img, txt2img, save_img
from lib.utils import ChatBot

@click.command()
@click.option('--story_prompt', prompt=True,
              default='Once upon a time, there was a little tiger named lua',
              help='llm prompt')
@click.option('--output_dir', prompt=True, default='output',
              help='output directory')
@click.option('--system_prompt_path', prompt=True, default='prompts/story_system_prompt1.md',
              help='system prompt path')
def main(story_prompt, output_dir, system_prompt_path):
    system_prompt = open(system_prompt_path).read()
    os.system(f'mkdir -p {output_dir}')
    chatbot = ChatBot(system_prompt)
    print('Querying the llm for story completion...')
    story_prompt = f"Children story to complete: \n---BEGIN_PROMPT---\n{story_prompt}\n---END_PROMPT---\n\nOutput:"
    response = chatbot(story_prompt)
    print('Response:\n', response)


    story_fn = os.path.join(output_dir, 'story.txt')
    with open(story_fn, 'w') as f:
        f.write(response)

    print(f'Done generating the story, saved in {story_fn}, starting to generate illustrations...')
    for i, story in enumerate(tqdm.tqdm([r for r in response.split('\n') if r.strip() != ''])):
        prompt = chatbot(f'Summarize the following paragraph you just created in one sentence, replace all references to the characters by their name: \n---BEGIN_PROMPT---\n{story}\n---END_PROMPT---\n\nOutput:')
        # append prompt to story_fn
        with open(story_fn, 'a') as f:
            f.write(f'\n\nprompt: {prompt}')
            
        prompt = story # TODO: currently the chatgpt prompt is unused, later we will introduce character generation to prompt using dreambooth
        img = txt2img(story, "worst-quality, watermark")
        save_img(img,
                 os.path.join(output_dir, f'{i}.png'))

if __name__ == '__main__':
    main()
