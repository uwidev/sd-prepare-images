#!/bin/env python

from imgutils.detect import booru_yolo, detection_visualize
from imgutils.tagging import wd14, overlap
from imgutils.upscale.cdc import upscale_with_cdc
from imgutils.restore import adversarial, nafnet, scunet
from pathlib import Path
from PIL import Image
import shutil
import os
import sys
from enum import Enum
import argparse
from matplotlib import pyplot as plt
from loguru import logger

IMG_EXT: list[str] = [".png", ".jpeg", ".jpg", ".webp"]
SCU: Path = Path("./workspace/scu/")
CROP: Path = Path("./workspace/crop/")
UP: Path = Path("./workspace/up/")
RAW: Path = Path("./raw/")
DONE: Path = Path("./done/")
WORKSPACE: list[Path] = [SCU, CROP, UP]


class LABEL(Enum):
    HEAD = "head"
    HCAT = "hcat"
    SHLD = "shld"
    BUST = "bust"
    BOOB = "boob"
    SIDEB = "sideb"
    BELLY = "belly"
    HIP = "hip"
    NOPAN = "nopan"
    BUTT = "butt"
    ASS = "ass"


class FRAMING(Enum):
    PORTRAIT = [[LABEL.HEAD, LABEL.HCAT], [LABEL.SHLD]]
    UPPER_BODY = [[LABEL.HEAD, LABEL.HCAT], [LABEL.BUST, LABEL.BOOB, LABEL.SIDEB]]
    COWBOY_SHOT = [
        [LABEL.HEAD, LABEL.HCAT],
        [LABEL.HIP, LABEL.NOPAN, LABEL.BUTT, LABEL.ASS],
    ]


# amount of heads down from bottom of head to center of body part
class SEARCH(Enum):
    SHLD = 0.5
    BUST = 1.0
    BOOB = 1.0
    SIDEB = 1.0
    BELLY = 1.5
    HIP = 1.75
    NOPAN = 1.75
    BUTT = 1.75
    ASS = 1.75


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


def search_overlapping_bbox(
    reference: list[int, int, int, int],
    bboxes: list[list[int, int, int, int]],
    search_multiplier: float = 1.2,
):
    """Find the closest bounding box given bounding boxes

    Returns the index and the bbox"""
    search_bbox = expand_bbox(reference, search_multiplier)

    for i, bbox in enumerate(bboxes):
        if is_bboxes_overlapping(search_bbox, bbox):
            return i, bbox

    return None, None


def expand_bbox(
    box,
    mult: float,
    *,
    contract_before: float = 0.0,
    left=True,
    up=True,
    right=True,
    down=True,
):
    """Expand a bounding box by a percent multiplier.

    You can specify the direction to expand. Optionally contracts the box
    before expanding, but it won't contract directions to be expanded.
    """
    width = box[2] - box[0]
    height = box[3] - box[1]
    left = int(left)
    up = int(up)
    right = int(right)
    down = int(down)

    # logger.debug("og width={}", width)
    logger.debug("height of head={}", height)

    _box = [
        int(box[0] + width * contract_before / 2 * (1 - left)),
        int(box[1] + height * contract_before / 2 * (1 - up)),
        int(box[2] - width * contract_before / 2 * (1 - right)),
        int(box[3] - height * contract_before / 2 * (1 - down)),
    ]

    # new_width = _box[2] - _box[0]
    # logger.debug("contracted width = {}", new_width)
    # logger.debug("real % shrunk = {}", (width - new_width) / width)
    logger.debug("contracted head={}", _box)

    return [
        int(_box[0] - width * mult * left),
        int(_box[1] - height * mult * up),
        int(_box[2] + width * mult * right),
        int(_box[3] + height * mult * down),
    ]


def is_bboxes_overlapping(b1, b2):
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


def create_head_body_bbox_pairs(
    yres: list[tuple[tuple[int, int, int, int], str, float]],
    threshold: float,
    *,
    crop_all: bool,
):
    """

    Given a list of (head) bboxes references, search for the closest
    expected bbox (preference) to the reference.
    """
    ybox, ylabel, yconf = zip(*yres)
    ybox: list[tuple[tuple[int, int, int, int]]]
    ylabel: list[str]
    yconf: list[float]

    logger.debug("finding matching part...")
    # each element of ybboxes supposedly belongs to one body
    ybboxes: list[list[list[int, int, int, int]]] = []

    # a list of indices that correspond to their location
    # in ybox, ylabel, yconf
    heads_index_ref = find_indices(LABEL.HCAT.value, ylabel) + find_indices(
        LABEL.HEAD.value, ylabel
    )
    if not heads_index_ref:
        logger.debug("no head or not good enough")
        return None

    heads = gather_bboxes(heads_index_ref, ybox, yconf, threshold)

    logger.debug("heads={}", heads)

    preference = [
        LABEL.SHLD,
        LABEL.BUST,
        LABEL.BOOB,
        LABEL.SIDEB,
        LABEL.BELLY,
        LABEL.HIP,
        LABEL.NOPAN,
        LABEL.BUTT,
        LABEL.ASS,
    ]

    for i, head in enumerate(heads):
        logger.debug("")
        logger.debug(">>> HEAD {}", head)
        dir_search = {
            "down": {"left": 0, "up": 0, "right": 0},
            "not_down": {"down": 0},
            "all": {},
        }
        # look down, if not work look left/right, otherwise look all
        for dir, dir_kwargs in dir_search.items():
            logger.debug("dir={}", dir)
            for prefer in preference:
                found = False
                logger.debug("looking for {}", prefer.value)
                prefer: LABEL
                bboxes_index_ref = find_indices(prefer.value, ylabel)
                bboxes = gather_bboxes(bboxes_index_ref, ybox, yconf)

                for j, bbox in enumerate(bboxes):
                    # check if this bbox confidence is above threshold
                    if yconf[bboxes_index_ref[j]] < threshold:
                        logger.debug("low confidence, continuing...")
                        continue

                    logger.debug("found bbox {}", bbox)
                    search_range = SEARCH[prefer.name].value
                    logger.debug("search_range={}", search_range)

                    head_ref_bbox_search = expand_bbox(
                        head, search_range, contract_before=0.5, **dir_kwargs
                    )
                    logger.debug("head_ref_bbox_search={}", head_ref_bbox_search)
                    closest = is_bboxes_overlapping(head_ref_bbox_search, bbox)

                    if closest:
                        logger.debug(f"found overlapping for {prefer.value}!")
                        ybboxes.append([head, bbox])
                        found = True
                        break

                    logger.debug("")

                logger.debug("")
                if found:
                    break

            logger.debug("")
            if found:
                break

        if not found or crop_all:
            logger.debug("not found, appending only head")
            ybboxes.append([head])

    logger.debug("ybboxes={}", ybboxes)
    return ybboxes


def find_indices(query: str, reference: list) -> [int]:
    """Find the indices where query exists in reference"""
    return [i for i, value in enumerate(reference) if value == query]


def gather_confs(indices: list[int], reference: list):
    """Return a list with only given indices"""
    return [reference[i] for i in indices]


def gather_bboxes(
    indices: list[int],
    bboxes: list[tuple[int, int, int, int]],
    confidences: list[float],
    threshold: float = 0.0,
):
    """Return bboxes whose confidences are above the threshold"""
    return [bboxes[i] for i in indices if confidences[i] > threshold]


def crop_dynamic(im_pth: Path, crop_all: bool = False) -> list[Path]:
    """Crop preferring portrait, upper body, then head
    Return a the path of the output"""
    logger.debug("starting crop on {}", im_pth.stem)
    if im_pth.suffix not in IMG_EXT:
        return

    ret: list = []
    out = CROP / f"{im_pth.stem}-crop.webp"
    if out.exists():
        logger.info("crop already exists for {}", im_pth.stem)
        return [out]

    # try yolo
    threshold = 0.3
    yres = booru_yolo.detect_with_booru_yolo(im_pth, "yolov8m_as03")

    # # debug show potential crops
    # plt.imshow(detection_visualize(im_pth, yres))
    # plt.show()

    raw_bboxes: list[list[list[int, int, int, int]]] = create_head_body_bbox_pairs(
        yres, threshold, crop_all=crop_all
    )

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
            #     logger.debug(out)
            #     logger.debug(set(og).difference(set(tags)))
            #     logger.debug('')
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
        logger.info("upscaling already exists for {}", im_pth.stem)
        return

    im: Image = Image.open(im_pth)
    px = im.width * im.height
    if px < 512 * 512:  # x2 if < 0.25MP
        im_up = upscale_with_cdc(im_pth, "HGSR-MHR_X2_1680")
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
        logger.info("scu restore already exists for {}", im_pth.stem)
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

    # TODO: let user decide what to keep
    # - keep head
    # - keep portrait
    # - keep upper body
    # - keep cowboy shot
    # - keep full body
    # - keep eyes
    # - keep with hands (maybe?)
    # ---
    # at the moment, only keeps portrait/upper body/cowboy shot, trashes head
    # if keep all, keeps both

    parser.add_argument(
        "--crop", action="store_true", help="crop images, keep only head"
    )
    parser.add_argument(
        "--crop-all", action="store_true", help="crop images, keep head + extended"
    )
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
    parser.add_argument("--debug", action="store_true", help="log level debug")

    args = parser.parse_args()

    # Remove sinks from logger
    logger.remove()
    # Normally only log INFO, WARNING, and SUCCESS to stdout
    if not args.debug:
        logger.add(
            sys.stdout,
            level="INFO",
            filter=lambda record: record["level"].name
            in ["INFO", "SUCCESS", "WARNING"],
        )

    # Otherwise, log INFO and DEBUG
    else:
        logger.add(
            sys.stdout,
            level="DEBUG",
            filter=lambda record: record["level"].name
            in ["DEBUG", "INFO", "SUCCESS", "WARNING"],
        )

    # Always log ERROR and above to stderr
    logger.add(sys.stderr, level="ERROR")

    # Begin
    for dir in WORKSPACE + [DONE, RAW]:
        dir.mkdir(exist_ok=True)

    if args.clean:
        logger.info("start clean")
        for pth in WORKSPACE:
            for file in pth.iterdir():
                os.remove(file)
        logger.success("done clean")
        exit()

    if args.restore or args.stage_1:
        logger.info("start restore")
        for pth in RAW.iterdir():
            restore_scu(pth)
        logger.success("done restore")

    if args.crop or args.stage_1:
        logger.info("start crop")
        for pth in SCU.iterdir():
            crop_dynamic(pth, args.crop_all)
        logger.success("done crop")
        logger.warning(
            "You should manually verify the crops before moving onto --stage-2."
        )

    if args.upscale or args.stage_2:
        logger.info("start upscale")
        for pth in CROP.iterdir():
            upscale(pth)

        for pth in RAW.iterdir():
            upscale(pth)

        logger.success("done upscale")

    if args.move or args.stage_2:
        logger.info("start move")
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

        logger.success("done move images")

    if args.tag:
        logger.info("start tag images")
        for pth in DONE.iterdir():
            tag_img(pth, drop_overlap=True)
        logger.success("done tag images")

    if args.tag_prepend:
        logger.info("start tag prepend")
        for pth in DONE.iterdir():
            prepend_tag(pth, args.tag_prepend)
        logger.success("done tag prepend")

    logger.success("done")


if __name__ == "__main__":
    main()
