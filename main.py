#!/bin/env python

from imgutils.detect import detect_heads, detection_visualize
from imgutils.tagging.wd14 import get_wd14_tags
from imgutils.upscale.cdc import upscale_with_cdc
from imgutils.restore import adversarial, nafnet, scunet
from pathlib import Path
from PIL import Image
import shutil
import os
import argparse

IMG_EXT: list[str] = [".png", ".jpeg", ".jpg", ".webp"]
SCU: Path = Path("./workspace/scu/")
HEAD: Path = Path("./workspace/head/")
UP: Path = Path("./workspace/up/")
RAW: Path = Path("./raw/")
DONE: Path = Path("./done/")
WORKSPACE: list[Path] = [SCU, HEAD, UP]


def pad_bbox(bbox: list[int], amount: int):
    return [
        bbox[0] - amount,
        bbox[1] - amount,
        bbox[2] + amount,
        bbox[3] + amount,
    ]


def exists_handler(p: Path) -> Path:
    """Return a renamed path if exists, otherwise return unchanged"""
    if not p.exists():
        return p

    name = p.stem
    ext = p.suffix

    number = 1
    if "_" in p.stem:
        try:
            name_split = name.split("_")
            number = int(name_split[-1])
            number += 1
            name = "_".join(name_split[:-1])
        except ValueError:
            pass

    return p.parent / f"{name}_{number}{ext}"


def crop_head(im_pth: Path) -> list[Path]:
    """Crop heads and return a the path of the output"""
    if im_pth.suffix not in IMG_EXT:
        return

    ret: list = []
    out = HEAD / f"{im_pth.stem}-head.webp"
    if out.exists():
        print(f"cropped head already exists for {im_pth.stem}")
        return [out]

    result = detect_heads(im_pth)

    im = Image.open(im_pth)
    for bbox, cropping_for, conf in result:
        out = exists_handler(out)
        crop = im.crop(bbox)
        crop.save(
            out,
            lossless=True,
            method=6,
            exact=True,
        )
        ret.append(out)

    return ret


def tag_img(im_pth: Path, delimiter: str = "\n") -> Path:
    """Tag image and return the written text file"""
    if im_pth.suffix not in IMG_EXT:
        return

    out = Path(f"./{im_pth.parent}/{im_pth.stem}.txt")
    if out.exists():
        print(f"tags already exists for {im_pth.stem}")
        return out

    ratings, general_tags, character_tags = get_wd14_tags(
        im_pth,
        "EVA02_Large",
        # general_mcut_enabled=True,
        # character_mcut_enabled=True,
        #
        # Mcut dynamically determines threshold as the point of max
        # difference. Might be useful for training concepts...? But not style
        # as it prunes too much.
    )

    rating = sorted(ratings.items(), key=lambda item: item[1], reverse=True)[0][0]

    tags = [rating] + list(general_tags.keys()) + list(character_tags.keys())

    write_tags(out, tags)

    return out


def upscale(im_pth: Path) -> Path:
    if im_pth.suffix not in IMG_EXT:
        return

    out = UP / f"{im_pth.stem}.webp"
    if out.exists():
        print(f"upscaling already exists for {im_pth.stem}")
        return

    im: Image = Image.open(im_pth)
    px = im.width * im.height
    if px < 768 * 768:  # x2
        im_up = upscale_with_cdc(im_pth)
        # if px < 1024 * 1024:
        #     im_up = upscale_with_cdc(im_pth, "HGSR-MHR_X2_1680")
        # else:  # x4
        #     im_up = upscale_with_cdc(im_pth)

        im_up.save(
            out,
            lossless=True,
            method=6,
            exact=True,
        )


def add_suffix_to_file(im_pth: Path, suffix: str) -> Path:
    return im_pth.parent / f"{im_pth.stem}{suffix}{im_pth.suffix}"


def restore_scu(im_pth: Path) -> Path:
    if im_pth.suffix not in IMG_EXT:
        return

    out = SCU / f"{im_pth.stem}.webp"
    if out.exists():
        print(f"scu restore already exists for {im_pth.stem}")
        return

    im: Image = Image.open(im_pth)

    scu: Image = scunet.restore_with_scunet(im)

    scu.save(out, lossless=True, method=6, exact=True)


def get_tags(pth: Path, delim="\n") -> list[str]:
    with pth.open("r") as fd:
        contents = fd.read()

    return contents.split(delim)


def write_tags(pth: Path, tags: list[str], delim="\n"):
    with open(pth, "w") as fd:
        fd.write(delim.join(tags))


def prepend_tag(pth: Path, tag: str, delim="\n"):
    if pth.suffix != ".txt":
        return

    tags = get_tags(pth, delim)
    if tag in tags:
        print(f"tag {tag} already exists for {pth}")
        return

    write_tags(pth, [tag] + tags, delim)


def main():
    parser = argparse.ArgumentParser(description="Preprocess images for AI training.")

    parser.add_argument("--clean", action="store_true", help="clean workspace")
    parser.add_argument("--restore", action="store_true", help="restore images")
    parser.add_argument("--crop", action="store_true", help="crop images")
    parser.add_argument("--upscale", action="store_true", help="upscale images")
    parser.add_argument(
        "--move",
        action="store_true",
        help="move finalized images and captions to ./done/",
    )
    parser.add_argument("--tag", action="store_true", help="tag images in ./done/")
    parser.add_argument("--tag-prepend", help="prepend tag to all captions in ./done/")

    parser.add_argument("--stage-1", action="store_true", help="restore and crop")
    parser.add_argument("--stage-2", action="store_true", help="upscale and move")

    args = parser.parse_args()

    for dir in WORKSPACE:
        dir.mkdir(exist_ok=True)

    if args.clean:
        for pth in WORKSPACE:
            for file in pth.iterdir():
                os.remove(file)
        exit()

    if args.restore or args.stage_1:
        for pth in RAW.iterdir():
            restore_scu(pth)

    if args.crop or args.stage_1:
        for pth in SCU.iterdir():
            crop_head(pth)
        print(
            "Finished cropping heads. You should manually verify the crops before moving onto --stage-2."
        )

    if args.upscale or args.stage_2:
        for pth in HEAD.iterdir():
            upscale(pth)

        for pth in RAW.iterdir():
            upscale(pth)

    if args.move or args.stage_2:
        for pth in SCU.iterdir():
            shutil.copy(pth, DONE / pth.name)

        for pth in HEAD.iterdir():
            shutil.copy(pth, DONE / pth.name)

        for pth in UP.iterdir():
            shutil.copy(pth, DONE / pth.name)

        # move captions if already exists
        for pth in RAW.iterdir():
            if pth.suffix == ".txt":
                shutil.copy(pth, DONE / pth.name)

    if args.tag:
        for pth in DONE.iterdir():
            tag_img(pth)

    if args.tag_prepend:
        for pth in DONE.iterdir():
            prepend_tag(pth, args.tag_prepend)


if __name__ == "__main__":
    main()
