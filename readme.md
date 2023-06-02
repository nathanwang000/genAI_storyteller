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
It depends on the automatic1111 api

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
the output video:

https://github.com/nathanwang000/genAI_storyteller/assets/5128093/5109cb44-3087-4eae-b659-2347d7899a0a

## (TODO) generating a story
## (TODO) story to images animation
## (TODO) training text inversion for toys and kids
## (TODO) text to video animation

TODO: text -> image -> image ... -> image

The image to image should ideally consider movement.
An simple way is to take an open source text to video model
and use Stable Diffusion to enhance the images
