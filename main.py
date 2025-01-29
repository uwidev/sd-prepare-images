#!/bin/env python

from imgutils.detect import booru_yolo, detection_visualize
from imgutils.tagging import wd14, overlap
from imgutils.upscale.cdc import upscale_with_cdc
from imgutils.restore import adversarial, nafnet, scunet
from pathlib import Path
from PIL import Image
import shutil
import os
import argparse
from matplotlib import pyplot as plt

IMG_EXT: list[str] = [".png", ".jpeg", ".jpg", ".webp"]
SCU: Path = Path("./workspace/scu/")
CROP: Path = Path("./workspace/crop/")
UP: Path = Path("./workspace/up/")
RAW: Path = Path("./raw/")
DONE: Path = Path("./done/")
WORKSPACE: list[Path] = [SCU, CROP, UP]


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
    out = CROP / f"{im_pth.stem}-head.webp"
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


def find_overlapping_bbox(
    reference: list[int, int, int, int],
    bboxes: list[list[int, int, int, int]],
    search_multiplier: float = 1.2,
):
    """Find the closest bounding box given bounding boxes

    Returns the index and the bbox"""
    # print(f"\nreference bbox {reference}")
    width = reference[2] - reference[0]
    height = reference[3] - reference[1]
    # print(f"{width=}")
    # print(f"{height=}")

    mult = search_multiplier - 1
    search_bbox = [
        int(reference[0] - width * mult),
        int(reference[1] - height * mult),
        int(reference[2] + width * mult),
        int(reference[3] + height * mult),
    ]
    # print(f"search bbox {search_bbox}\n")

    for i, bbox in enumerate(bboxes):
        if is_bboxes_overlapping(search_bbox, bbox):
            return i, bbox

    return None, None


def is_bboxes_overlapping(b1, b2):
    # print(f"{b1=}")
    # print(f"{b2=}")

    # Unpack the coordinates
    x1_1, y1_1, x2_1, y2_1 = b1
    x1_2, y1_2, x2_2, y2_2 = b2

    # Check if one box is to the left of the other
    if x2_1 < x1_2 or x2_2 < x1_1:
        return False

    # Check if one box is above the other
    if y2_1 < y1_2 or y2_2 < y1_1:
        return False

    return True


def crop_dynamic(im_pth: Path) -> list[Path]:
    """Crop preferring portrait, upper body, then head
    Return a the path of the output"""
    if im_pth.suffix not in IMG_EXT:
        return

    ret: list = []
    out = CROP / f"{im_pth.stem}-crop.webp"
    if out.exists():
        print(f"cropped head already exists for {im_pth.stem}")
        return [out]

    # try yolo
    threshold = 0.5
    yres = booru_yolo.detect_with_booru_yolo(im_pth, "yolov8m_as03")

    # # debug show potential crops
    # plt.imshow(detection_visualize(im_pth, yres))
    # plt.show()

    ybox, ylabel, yconf = zip(*yres)
    ybox: list[tuple[tuple[int, int, int, int]]]
    ylabel: list[str]
    yconf: list[float]

    def lbox(label: str) -> [int]:
        return ybox[ylabel.index(label)]

    def lconf(label: str) -> float:
        return yconf[ylabel.index(label)]

    def find_indices(label: str) -> [int]:
        # find the indices where label exists in in yolo label list
        return [i for i, ylabel in enumerate(ylabel) if ylabel == label]

    def gather_confs(indices: list[int]):
        return [yconf[i] for i in indices]

    def gather_bboxes(indices: list[int]):
        return [ybox[i] for i in indices]

    # TODO: If there are multiple heads, find the closest bbox within some
    # percentage of the head bbox These bboxes are assumed to be a pair.
    def find_valid_yolo_bboxes():
        """Find the two pairs of valid bboxes"""
        # each element of ybboxes supposedly belongs to one body
        ybboxes: list[list[list[int, int, int, int]]] = []

        head_inds = find_indices("head")
        if not head_inds:
            # print("no head or not good enough")
            return None

        heads = gather_bboxes(head_inds)
        ybboxes.extend([head] for head in heads)

        # head+shoulder
        # TODO: wasted compute, looks at all heads even if they are
        shlds = gather_bboxes(find_indices("shld"))
        for shld in shlds:
            if "shld" in ylabel:
                i, closest = find_overlapping_bbox(shld, heads)

                if closest:
                    if len(ybboxes[i]) == 1:  # head has no pair yet
                        # print("adding shoulder pair")
                        ybboxes[i].append(shld)

            if all(len(body) == 2 for body in ybboxes):
                return ybboxes

        # head+bust
        # TODO: wasted compute, looks at all heads even if they are
        busts = gather_bboxes(find_indices("bust"))
        for bust in busts:
            if lconf("bust") > threshold:
                i, closest = find_overlapping_bbox(bust, heads)

                if closest:
                    if len(ybboxes[i]) == 1:  # head has no pair yet
                        # print("adding bust pair")
                        ybboxes[i].append(bust)

            if all(len(body) == 2 for body in ybboxes):
                return ybboxes

        # head+boob
        # TODO: wasted compute, looks at all heads even if they are
        boobs = gather_bboxes(find_indices("boob"))
        # already paired
        for boob in boobs:
            if "boob" in ylabel and lconf("boob") > threshold:
                i, closest = find_overlapping_bbox(boob, heads)

                if closest:
                    if len(ybboxes[i]) == 1:  # head has no pair yet
                        # print("adding boob pair")
                        ybboxes[i].append(boob)

            if all(len(body) == 2 for body in ybboxes):
                return ybboxes

        # head+sideb
        # TODO: wasted compute, looks at all heads even if they are
        sidebs = gather_bboxes(find_indices("sideb"))
        for sideb in sidebs:
            if "sideb" in ylabel and lconf("sideb") > threshold:
                i, closest = find_overlapping_bbox(sideb, heads)

                if closest:
                    if len(ybboxes[i]) == 1:  # head has no pair yet
                        # print("adding sideb pair")
                        ybboxes[i].append(sideb)
            if all(len(body) == 2 for body in ybboxes):
                return ybboxes

        # head
        return ybboxes

    raw_bboxes: list[list[list[int, int, int, int]]] = find_valid_yolo_bboxes()

    # no bounding boxes found
    if not raw_bboxes:
        return

    def combine_bboxes(bboxes):
        return [
            min(box[0] for box in bboxes),
            min(box[1] for box in bboxes),
            max(box[2] for box in bboxes),
            max(box[3] for box in bboxes),
        ]

    bboxes = []
    for bbx in raw_bboxes:
        bboxes.append(combine_bboxes(bbx))

    im = Image.open(im_pth)

    for bbox in bboxes:
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


def tag_img(im_pth: Path, delim: str = ",", drop_overlap=False) -> Path:
    """Tag image and return the written text file"""
    if im_pth.suffix not in IMG_EXT:
        return

    out = Path(f"./{im_pth.parent}/{im_pth.stem}.txt")
    if out.exists():
        # if tag already exists, just do some cleanup?
        if drop_overlap:
            og = get_tags(out, delim)
            tags = overlap.drop_overlap_tags(og)
            # if og != tags:
            #     print(out)
            #     print(set(og).difference(set(tags)))
            #     print()
            write_tags(out, tags)
    else:
        ratings, general_tags, character_tags = wd14.get_wd14_tags(
            im_pth,
            "EVA02_Large",
            no_underline=True,
            drop_overlap=True,
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


def get_tags(pth: Path, delim=",") -> list[str]:
    with pth.open("r") as fd:
        contents = fd.read()

    return contents.split(delim)


def write_tags(pth: Path, tags: list[str], delim=","):
    with pth.open("w") as fd:
        fd.write(delim.join(tags))


def prepend_tag(pth: Path, tag: str, delim=","):
    if pth.suffix != ".txt":
        return

    tags = get_tags(pth, delim)
    if tag in tags:
        tags.remove(tag)

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

    for dir in WORKSPACE + [DONE, RAW]:
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
            crop_dynamic(pth)
        print(
            "Finished cropping heads. You should manually verify the crops before moving onto --stage-2."
        )

    if args.upscale or args.stage_2:
        for pth in CROP.iterdir():
            upscale(pth)

        for pth in RAW.iterdir():
            upscale(pth)

    if args.move or args.stage_2:
        for pth in SCU.iterdir():
            shutil.copy(pth, DONE / pth.name)

        for pth in CROP.iterdir():
            shutil.copy(pth, DONE / pth.name)

        for pth in UP.iterdir():
            shutil.copy(pth, DONE / pth.name)

        # move captions if already exists
        for pth in RAW.iterdir():
            if pth.suffix == ".txt":
                shutil.copy(pth, DONE / pth.name)

    if args.tag:
        for pth in DONE.iterdir():
            tag_img(pth, drop_overlap=True)

    if args.tag_prepend:
        for pth in DONE.iterdir():
            prepend_tag(pth, args.tag_prepend)


if __name__ == "__main__":
    main()
