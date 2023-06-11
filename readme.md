# genAI storyteller

This is a project with Hao and Jeeheh.

Our goal is to create storybooks generated using generative AI models
- LLMs for story text generation
- Stable Diffusion for illustrations (either picture or video)
- (Optional) Text to speech narration of the story

We also plan to support
- putting custom toys/kid into story teller using text inversion

## video to video animation

Currently we have a utility to create animation video from a driving video.
It depends on the automatic1111 [api](https://github.com/Mikubill/sd-webui-controlnet/wiki/API#integrating-sdapiv12img):

on mac to start a server, use
```bash webui.sh --no-half --api```

then generate videos as follows
```
python video2video.py
python create_video_from_images.py
```

The source/driving video:

https://github.com/nathanwang000/genAI_storyteller/assets/5128093/7580440a-b90a-4ce4-8f7c-196569eaefcf

With the prompt of "tiger" and negative prompt of "worst-quality",
the output video when using txt2img:

https://github.com/nathanwang000/genAI_storyteller/assets/5128093/5109cb44-3087-4eae-b659-2347d7899a0a

the output video when using img2img:

https://github.com/nathanwang000/genAI_storyteller/assets/5128093/04074a81-566b-4e49-b002-ed8e0955c56e

## generating a story with illustrations

```
python story2illustration.py
```
Currently doesn't handle negative prompt. Upon some thought, I think story creation should be interactive: that is have a chat with chatgpt to collaboratively create the story and then use stable diffusion to illustrate the story.


Prompt: generate a kid story about a tiger

ChatGPT response with Stable Diffusion illustration (breakdomainrealistic_M2050.safetensors [4f0d8aacd3] with negative prompt of "worst-quality"):

Once upon a time, in a lush green forest, there lived a friendly and curious tiger named Tilly. Tilly was unlike any other tiger in the forest. Instead of being fierce and intimidating, she had a gentle nature and a heart full of kindness.

![00000-2002325985](https://github.com/nathanwang000/genAI_storyteller/assets/5128093/6e94691e-0092-4d45-a273-7f285d07781c)

Tilly loved exploring the forest and meeting new friends. She would spend her days hopping from one tree branch to another, carefully observing the vibrant colors of the forest and the playful antics of the animals around her.

![00001-3571555173](https://github.com/nathanwang000/genAI_storyteller/assets/5128093/75f2a1f9-c43f-4697-b236-d670b5f7b927)

One sunny morning, as Tilly was strolling near a sparkling river, she heard a faint cry for help. She followed the sound and discovered a little squirrel named Sammy, stuck on a tall tree branch.

![00002-4015718530](https://github.com/nathanwang000/genAI_storyteller/assets/5128093/82c437e2-66e4-4430-9067-d9727a491a9e)

Without hesitation, Tilly sprang into action. She carefully climbed up the tree, using her strong claws, and gently rescued Sammy. From that moment on, Tilly and Sammy became inseparable friends.

![00003-929176501](https://github.com/nathanwang000/genAI_storyteller/assets/5128093/c8c9d0d3-3170-4d0f-889a-130545ea4c2f)

News of Tilly's kind-hearted nature spread throughout the forest, and animals from far and wide sought her help. Whenever someone was in trouble or needed guidance, they would approach Tilly, knowing she would offer a helping paw.

![00004-3007007950](https://github.com/nathanwang000/genAI_storyteller/assets/5128093/4727b81a-8c60-4c0e-8974-c8ebca15fb28)

One day, a group of lost baby birds found their way to Tilly. They were tired and hungry, unable to find their way back to their nest. Tilly comforted them with her warm presence and led them safely back home. The grateful bird parents chirped joyfully and thanked Tilly for her compassion.

![00005-3348461783](https://github.com/nathanwang000/genAI_storyteller/assets/5128093/db47fad7-5766-4dce-b8b8-6e0b87df9466)

As time passed, Tilly's reputation as a kind and helpful tiger grew. Animals from neighboring forests would come seeking her assistance, and Tilly never turned them away. She dedicated herself to protecting the weak, teaching the young, and spreading love and harmony among all creatures.

![00006-405008219](https://github.com/nathanwang000/genAI_storyteller/assets/5128093/2e740d59-eab8-45ca-b2a1-0ddd5acf90ce)

The forest became a peaceful haven, thanks to Tilly's unwavering commitment to kindness. Animals of all kinds lived together in harmony, sharing their stories and celebrating their differences.

![00007-1948873742](https://github.com/nathanwang000/genAI_storyteller/assets/5128093/e466769d-5de4-459b-8202-c1badc94fa5f)

And so, the story of Tilly, the gentle tiger, became a legend in the forest. Her acts of compassion and bravery were passed down from generation to generation, reminding all the animals of the power of kindness and the importance of helping others.

![00008-3631239971](https://github.com/nathanwang000/genAI_storyteller/assets/5128093/d61f207a-e8dd-4f11-8716-b77e03d1b499)

And as long as there were tigers like Tilly, the forest remained a place where friendship, love, and acceptance flourished, creating a beautiful world for everyone to enjoy.

![00009-1509688215](https://github.com/nathanwang000/genAI_storyteller/assets/5128093/256ca9b2-cf0f-4e01-a3a5-fb465ab0ba9d)

## (TODO) training text inversion for toys and kids
## (TODO) text to video animation

TODO: text -> image -> image ... -> image

The image to image should ideally consider movement.
An simple way is to take an open source text to video model
and use Stable Diffusion to enhance the images

## (in progress) aligning faces

use the function in lib/utils.py to download some images
```
download_ddgs_image_search('jamal crawford')
```

then align images with
```
python align_faces.py -d output/ddgs_images/jamal_crawford/1.jpeg -i output/ddgs_images/jamal_crawford -o output/jamal_crawford
```
