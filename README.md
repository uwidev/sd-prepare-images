## Overview
This is a python script that utilizes [dghs/imgutils](https://github.com/deepghs/imgutils) for preparing images to be trained for stable diffusion models (or similar). It's somewhat opinionated and is for anime style trainings, but could be extended for other forms of training.

## Project Structure
Images to-be-processed should be placed in `./raw/`. You can also place captions in here as well; if they exist, these images will not be tagged when the flag is set (may change in the future, I currently [pre-tag all images in Hydrus](https://github.com/uwidev/wd-e621-hydrus-tagger)).

Finalied images are moved to `./done/` in a way that ensures the best quality and upscaled images, ignoring inferior images. Tagging will be done based on this directory.

## Usage
```
usage: main.py [-h] [--clean] [--restore] [--crop] [--upscale] [--move]
               [--tag] [--tag-prepend TAG_PREPEND] [--stage-1] [--stage-2]

Preprocess images for AI training.

options:
  -h, --help            show this help message and exit
  --clean               clean workspace
  --restore             restore images
  --crop                crop images
  --upscale             upscale images
  --move                move finalized images and captions to ./done/
  --tag                 tag images in ./done/
  --tag-prepend TAG_PREPEND
                        prepend tag to all captions in ./done/
  --stage-1             restore and crop
  --stage-2             upscale and move
```

Note that the order of operations internally is as folows...
```
restore (denoise jpeg artifacts) -> crop -> upscale -> tag -> tag-prepend
```
Not including a flag will not do that step.

### Why CLI has multiple steps...
Sometimes the cropper may crop false-positives, or maybe a face is just way too small to be upscaled with reasonable quality. To save time and compute power, do `--stage-1`, then go to `./workspace/crop/` and manually delete anything you don't want to be further procesed. Afterwards, you can continue to `--stage-2`.

### About `--tag-prepend`
To prepend the artist/style/concept tag to all caption files, if exists. It will not add the tag if it already exists.

### About `--clean`
Between datasets, you should clean the workspace, otherwise it will also process the previous dataset.
