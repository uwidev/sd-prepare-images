#!/bin/env python

from collections.abc import Iterable, Sequence, Mapping
from imgutils.detect import booru_yolo, person
from imgutils.tagging import wd14, overlap, sort_tags, order
from imgutils.upscale.cdc import upscale_with_cdc
from imgutils.restore import scunet
from pathlib import Path
from PIL import Image
from PIL.ImageFile import ImageFile
import shutil
import os
import sys
from enum import Enum
import argparse
from loguru import logger
from itertools import combinations, product
import csv
from transformers import AutoTokenizer, PreTrainedTokenizerBase

IMG_EXT: list[str] = [".png", ".jpeg", ".jpg", ".webp"]
RAW: Path = Path("./raw/")
CROP: Path = Path("./workspace/crop/")
DOWNSCALE: Path = Path("./workspace/downscale/")
SCUTODO: Path = Path("./workspace/scu-todo/")
SCU: Path = Path("./workspace/scu/")
UP: Path = Path("./workspace/up/")
DONE: Path = Path("./done/")
WORKSPACE: list[Path] = [SCU, CROP, UP, DOWNSCALE]

ALWAYS_BLACKLIST = [
	"virtual youtuber",
	"borrowed character",
	"dual persona",
]

type BBox = tuple[int, int, int, int]
type LB = tuple[LABEL, BBox]


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
	SPLIT = "split"
	VSPLIT = "vsplit"
	VSPRD = "vsprd"
	JACKO = "jacko"
	JACKX = "jackx"


class FRAMING(Enum):
	PORTRAIT = ((LABEL.HEAD, LABEL.HCAT), [LABEL.SHLD])
	UPPER_BODY = ((LABEL.HEAD, LABEL.HCAT), [LABEL.BUST, LABEL.BOOB, LABEL.SIDEB])
	COWBOY_SHOT = (
		(LABEL.HEAD, LABEL.HCAT),
		(LABEL.HIP, LABEL.NOPAN, LABEL.BUTT, LABEL.ASS),
	)


class CLIP_RESOLVE(Enum):
	SMART = "smart"
	DROP = "drop"
	ADD = "add"


# the relative distance from the previous part
RELATIVE_DISTANCE = [
	((LABEL.HEAD, LABEL.HCAT), 0.0),
	((LABEL.SHLD,), 0.5),
	((LABEL.BUST, LABEL.BOOB, LABEL.SIDEB), 0.5),
	((LABEL.BELLY,), 0.5),
	((LABEL.HIP,), 0.25),
	(
		(
			LABEL.NOPAN,
			LABEL.BUTT,
			LABEL.ASS,
			LABEL.SPLIT,
			LABEL.VSPLIT,
			LABEL.VSPRD,
			LABEL.JACKO,
			LABEL.JACKX,
		),
		0.25,
	),
]

DIRS = [  # index - 2 is reverse direction
	{  # down
		"left": False,
		"up": False,
		"right": False,
		"contract_before": 0.5,
	},
	{  # left
		"up": False,
		"right": False,
		"down": False,
		"contract_before": 0.5,
	},
	{  # up
		"left": False,
		"right": False,
		"down": False,
		"contract_before": 0.5,
	},
	{  # right
		"left": False,
		"up": False,
		"down": False,
		"contract_before": 0.5,
	},
]


def load_danbooru_tags(
	*, use_spaces: bool = True
) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
	"""Return danbooru tags in format ((ratings), (tags), (characters))"""
	ratings: list[str] = []
	tags: list[str] = []
	characters: list[str] = []

	with Path("./selected_tags.csv").open("r") as fp:
		reader = csv.DictReader(fp)
		for row in reader:
			if row["category"] == "9":
				ratings.append(row["name"])
			elif row["category"] == "0":
				tags.append(row["name"])
			elif row["category"] == "4":
				characters.append(row["name"])

	if use_spaces:
		ratings = [rating.replace("_", " ") for rating in ratings]
		tags = [tag.replace("_", " ") for tag in tags]
		characters = [character.replace("_", " ") for character in characters]

	return tuple(ratings), tuple(tags), tuple(characters)


def get_distance(l1: LABEL, l2: LABEL) -> float:
	"""Return the distance between two body parts in reference to head size"""
	l1_index = None
	l2_index = None
	for i, labels_distance in enumerate(RELATIVE_DISTANCE):
		labels, _ = labels_distance

		if l1 in labels:
			l1_index = i

		if l2 in labels:
			l2_index = i

	if None in [l1_index, l2_index]:
		msg = f"Labels {l1} or {l2} are not registered distances!"
		raise KeyError(msg)

	return sum(pair[1] for pair in RELATIVE_DISTANCE[l1_index : l2_index + 1])  # pyright: ignore[reportOptionalOperand]


def pad_bbox(bbox: list[int], amount: int):
	return [
		bbox[0] - amount,
		bbox[1] - amount,
		bbox[2] + amount,
		bbox[3] + amount,
	]


def exists_handler(p: Path) -> Path:
	"""Return a renamed path if exists, otherwise return unchanged"""
	name = p.stem
	ext = p.suffix

	def increment_rename(p: Path, num: int = 0):
		num += 1
		new_path = p.parent / f"{name}_{num}{ext}"

		if not p.exists():
			return p

		return increment_rename(new_path, num)

	return increment_rename(p)


def search_overlapping_bbox(
	reference: LB,
	bboxes: LB,
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
	# logger.debug("height of head={}", height)

	_box = (
		int(box[0] + width * contract_before / 2 * (1 - left)),
		int(box[1] + height * contract_before / 2 * (1 - up)),
		int(box[2] - width * contract_before / 2 * (1 - right)),
		int(box[3] - height * contract_before / 2 * (1 - down)),
	)

	# new_width = _box[2] - _box[0]
	# logger.debug("contracted width = {}", new_width)
	# logger.debug("real % shrunk = {}", (width - new_width) / width)
	# logger.debug("contracted head={}", _box)

	return (
		int(_box[0] - width * mult * left),
		int(_box[1] - height * mult * up),
		int(_box[2] + width * mult * right),
		int(_box[3] + height * mult * down),
	)


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


def flatten_label_bboxes(
	labels_bboxes: Iterable[tuple[LABEL, tuple[BBox, ...]]],
) -> tuple[tuple[LB, ...], ...]:
	"""Return a re-organized structure to prepare for Cartesian product.

	i.e    label A        | label B    | label C
	(((lA, b1), (la, b2)) | ((lB, b1)) | ((lC, b1)))

	would result in ideally result in...
	lAb1-lBb1, lAb2-lBb1, lAb1-lCb1, lAb2-lCb2

	The return structure groups label-bbox pairs in a single tuple. If there
	are multiple labels, there is expected to be multiple groupings. In other
	words:
	((groupings of head), (groupings of chest), ...)
	"""
	return tuple(
		tuple(
			(
				label,
				bbox,
			)
			for bbox in bboxes
		)
		for label, bboxes in labels_bboxes
	)


def create_search_bboxes(
	label_bbox_combo: tuple[LB, ...], dir: int, dir_kwargs: dict
) -> tuple[LB, ...]:
	ret: list[LB] = []
	for i, label_bbox in enumerate(label_bbox_combo[:-1]):
		label, bbox = label_bbox
		label_next = label_bbox_combo[i + 1][0]
		distance = get_distance(label, label_next)

		logger.debug("expanding by distance {}", distance)

		ret.append((label, expand_bbox(label_bbox[1], distance, **dir_kwargs)))

	# last bbox expands reverse direction
	_dir = dir - 2
	label, bbox = label_bbox_combo[-1]
	# label_prev = label_bbox_combo[-2][0]
	# distance = get_distance(label_prev, label)
	distance = 0

	# -1 is all directions, use dir_kwargs (which should be empty for this case)
	dir_kwargs = DIRS[_dir] if dir != -1 else dir_kwargs
	ret.append((label, expand_bbox(bbox, distance, **dir_kwargs)))

	logger.debug("expanding by distance {}", distance)
	logger.debug("bboxes expanded for search {}", ret)
	return tuple(ret)


def is_bboxes_connected(
	search_labels_bboxes: Sequence[LB],
) -> bool:
	"""Return True if A<->B<->...<->Z"""

	# TODO: naive implementation
	# if four bboxes are arranged in a square, it's possible that AB connect
	# and CD connect, but nothing in between, meaning they're not actually connected
	# see /home/timmy/in/graph_test/connection_graph.py

	already_overlaps = [False] * len(search_labels_bboxes)
	for i, label_bbox in enumerate(search_labels_bboxes):
		if already_overlaps[i]:
			continue

		_, this_bbox = label_bbox
		other_bboxes = [*search_labels_bboxes[0:i], *search_labels_bboxes[i + 1 :]]

		# if a bbox does not overlap with at least one bbox
		# then all bboxes are not connected
		for j, other_label_bbox in enumerate(other_bboxes):
			_, other_bbox = other_label_bbox

			if is_bboxes_overlapping(this_bbox, other_bbox):
				already_overlaps[i] = True
				# adjust j cause list no longer has i
				_j = j if j < i else j + 1
				already_overlaps[_j] = True

			else:
				return False

	return True


def find_crops(
	yres: Sequence[tuple[BBox, str, float]],
	threshold: float,
	*,
	labels_to_crop: Sequence[list[LABEL]],
	rollup: bool = False,
	prefer_largest: bool = False,
) -> set[BBox]:
	"""Find valid crops in an image given things to crop"""
	ybox, ylabel, yconf = zip(*yres)
	ybox: tuple[BBox, ...]
	ylabel: tuple[str, ...]
	yconf: tuple[float, ...]

	# list of bboxes to crop
	crops: set[BBox] = set()

	# order to_crop so bboxes connect top-down
	logger.debug("to crop {}", labels_to_crop)
	to_crop_ordered: list[list[LABEL]] = []
	for labels in labels_to_crop:
		if len(labels) == 1:
			to_crop_ordered.append(labels)
		else:
			to_crop_ordered.append([label for label in LABEL if label in labels])

	logger.debug("to_crop re-ordered as {}", to_crop_ordered)

	# e.g. to_crop = [[head], [head, hip]]
	for labels in to_crop_ordered:
		# labels: tuple[LABEL]
		labels_bboxes: list[tuple[LABEL, tuple[BBox, ...]]] = []

		for label in labels:
			label_indices = find_indices(label.value, ylabel)
			bboxes = gather_bboxes(label_indices, ybox, yconf, threshold)
			if bboxes:
				labels_bboxes.append((label, bboxes))

		if len(labels_bboxes) == 0:  # no bboxes found corresponding to labels
			continue

		# if we're only cropping for one label, just return the bboxes
		if len(labels) == 1:
			for label, bbox in labels_bboxes:
				logger.debug("only one label {}, adding bboxes {}", label, bboxes)  # pyright: ignore[reportPossiblyUnboundVariable]
				crops.update(bbox)
			continue

		# e.g. grouped_bboxes = [[top left, top right], [bottom left, bottom right]]

		# there may be >1 bbox for a label
		# for each combination of bboxes of one label to the others,
		# it's valid if all bboxes overlap; add to finalized_crop

		# flatten label_bboxes from [(LABEL, [[bbox]])] to [[[LABEL, bbox]]]
		# where the first-depth index is the label group corresponding to labels
		labels_bboxes_flattened = flatten_label_bboxes(labels_bboxes)

		debug_labels_bboxes = [
			[label, [i for i, _ in enumerate(bboxes)]]
			for label, bboxes in labels_bboxes
		]
		debug_labels_bboxes_flattened = flatten_label_bboxes(debug_labels_bboxes)  # pyright: ignore[reportArgumentType]

		logger.debug("label_bboxes={}", labels_bboxes)
		logger.debug("debug_label_bboxes={}", debug_labels_bboxes)
		logger.debug(
			"debug_labels_bboxes flattened to {}", debug_labels_bboxes_flattened
		)

		labels_bboxes_combinations = []
		debug_labels_bboxes_combinations = []

		if not rollup:
			labels_bboxes_combinations.extend(product(*labels_bboxes_flattened))
			debug_labels_bboxes_combinations.extend(
				product(*debug_labels_bboxes_flattened)
			)

		if rollup:
			logger.debug("doing crop rollup")
			for rollup_len in range(len(labels_bboxes_flattened), 0, -1):
				labels_bboxes_combinations.extend(
					product(*labels_bboxes_flattened[:rollup_len])
				)
				debug_labels_bboxes_combinations.extend(
					product(*debug_labels_bboxes_flattened[:rollup_len])
				)

		logger.debug("all combinations: {}", labels_bboxes_combinations)
		logger.debug("all combinations (debug): {}", debug_labels_bboxes_combinations)
		logger.debug("combination length = {}", len(debug_labels_bboxes_combinations))

		# largest bbox combo we already found
		largest_found: list[tuple[tuple[LABEL, tuple[int,]],]] = []
		for label_bbox_combo in labels_bboxes_combinations:
			label_bbox_combo: tuple[LB]
			# e.g. combination = [(head, top_left), (hip, bottom_left)]

			found_largest = False
			if prefer_largest:
				logger.debug("already found valid crops {}", crops)
				logger.debug(
					"checking if {} is a subset of any valid crops", label_bbox_combo
				)
				for found in largest_found:
					# if [head, shld] a subset of [head, shld, belly]
					if set(label_bbox_combo).issubset(found):
						found_largest = True
						break

			if found_largest:
				logger.info("found larger existing crop, skipping...")
				continue

			logger.debug("=" * 64)
			logger.debug("finding bbox overlaps for combo {}", label_bbox_combo)

			# rollup will include the original top as a combo, and only the top
			if len(label_bbox_combo) == 1:
				crops.add(label_bbox_combo[0][1])
				continue

			# find the combination where all bboxes overlap (by inverse logic)
			# search down first, then left, up, right, then all (at end)

			is_connected = False
			for dir, dir_kwargs in enumerate(DIRS):
				logger.debug("expanding search in direction {}", dir)
				logger.debug("dir_kwargs {}", dir_kwargs)

				expanded_search_bboxes = create_search_bboxes(
					label_bbox_combo, dir, dir_kwargs
				)

				is_connected = is_bboxes_connected(expanded_search_bboxes)
				if is_connected:
					if prefer_largest:
						largest_found.append(label_bbox_combo)
					combo_bboxes = tuple(lb[1] for lb in label_bbox_combo)
					logger.debug("combo found, adding {}", combo_bboxes)
					crops.add(combine_bboxes(combo_bboxes))
					break

			# lastly, try searching all directions
			if not is_connected:
				expanded_search_bboxes = create_search_bboxes(label_bbox_combo, -1, {})
				if is_bboxes_connected(expanded_search_bboxes):
					if prefer_largest:
						largest_found.append(label_bbox_combo)
					combo_bboxes = tuple(lb[1] for lb in label_bbox_combo)
					logger.debug("combo found, adding {}", combo_bboxes)
					crops.add(combine_bboxes(combo_bboxes))

	return crops


def find_indices(query: str, reference: tuple) -> tuple[int, ...]:
	"""Find all indices where query exists in reference

	Returns a list of indices where the match was found.
	"""
	return tuple(i for i, value in enumerate(reference) if value == query)


def gather_confs(indices: list[int], reference: list):
	"""Return a list with only given indices"""
	return [reference[i] for i in indices]


def gather_bboxes(
	indices: tuple[int, ...],
	bboxes: tuple[BBox, ...],
	confidences: tuple[float, ...],
	threshold: float = 0.0,
) -> tuple[BBox, ...]:
	"""Return bboxes whose confidences are above the threshold"""
	return tuple(bboxes[i] for i in indices if confidences[i] > threshold)


def parse_crops(crops: str) -> list[list[LABEL]]:
	ret = []

	for crop in crops:
		if crop == "all":
			ret.append(list(LABEL))
			continue

		combo = crop.split("+")
		ret.append([LABEL(c) for c in combo])

	return ret


def combine_bboxes(bboxes) -> BBox:
	return (
		min(box[0] for box in bboxes),
		min(box[1] for box in bboxes),
		max(box[2] for box in bboxes),
		max(box[3] for box in bboxes),
	)


def crop_dynamic(
	im_pth: Path,
	crops: list[list[LABEL]],
	*,
	rollup: bool = False,
	prefer_largest: bool = False,
) -> list[Path] | None:
	"""Crop preferring portrait, upper body, then head

	Return a the path of the output"""
	# TODO: current implementation does not know how and if a bbox belongs to
	# a person, use yolo in conjunction with another models to determine if a
	# bbox belongs to that person
	# --
	# TODO: take a look at detect similarity...

	if im_pth.suffix not in IMG_EXT:
		return None

	logger.info("starting crop on {}", im_pth.stem)

	# # debug show potential crops
	# yres = booru_yolo.detect_with_booru_yolo(im_pth, "yolov8m_as03")
	# plt.imshow(detection_visualize(im_pth, yres))
	# plt.show()
	# # return

	ret: list = []
	out = CROP / f"{im_pth.stem}-crop.webp"
	if out.exists():
		out.touch(exist_ok=True)
		logger.info("crop already exists for {}", im_pth.stem)
		return [out]

	yres = booru_yolo.detect_with_booru_yolo(im_pth, "yolov8m_as03")
	if len(yres) == 0:  # no detections
		return

	threshold = 0.3
	raw_bboxes: set[BBox] = find_crops(
		yres,
		threshold,
		labels_to_crop=crops,
		rollup=rollup,
		prefer_largest=prefer_largest,
	)

	logger.info("found {} crops", len(raw_bboxes))

	# no bounding boxes found
	if not raw_bboxes:
		return

	# bboxes = []
	# for bbx in raw_bboxes:
	#     logger.debug("bbx={}", bbx)
	#     bboxes.append(combine_bboxes(bbx))

	bboxes = raw_bboxes

	logger.debug("finalized bboxes for image {}", bboxes)

	im = Image.open(im_pth)

	for bbox in bboxes:
		out = exists_handler(out)
		crop = im.crop(bbox)
		crop.save(
			out,
			lossless=True,
			method=0,
			quality=0,
		)
		ret.append(out)

	return ret


def generate_cartesian_crops(
	yres: Iterable[tuple[BBox, str, float]],
	threshold: float,
	*,
	labels_to_crop: Iterable[list[LABEL]],
	prefer_largest: bool = False,
) -> set[tuple[tuple[LABEL, ...], BBox]]:
	"""Generate the combation of crops, then the cartesian product of them.

	In an example of label-bbox pairs...
	A1 A2 B1 B2

	The resulting combo of crop should be as follows.
	A1+B1 A1+B2 A2+B1 A2+B2

	This is not as trivial as just calling product(), We need to do
	combination() first when considering rollup (always). That is, we need to
	find crops on combinations of label size n, n-1, n-2, ..., 1.
	"""
	ybox, ylabel, yconf = zip(*yres)
	ybox: tuple[BBox, ...]
	ylabel: tuple[str, ...]
	yconf: tuple[float, ...]

	# set of bboxes to crop (prevent duplicate bboxes)
	unique_crops: set[BBox] = set()

	# actual return thingy
	crops: set[tuple[tuple[LABEL, ...], BBox]] = set()

	for labels in labels_to_crop:
		labels_bboxes: list[tuple[LABEL, tuple[BBox, ...]]] = []

		# gather bboxes according to labels user asked
		for label in labels:
			label_indices = find_indices(label.value, ylabel)
			bboxes = gather_bboxes(label_indices, ybox, yconf, threshold)
			if bboxes:
				labels_bboxes.append((label, bboxes))

		# no bboxes found corresponding to labels
		if len(labels_bboxes) == 0:
			continue

		# if only cropping for one label, return bboxes for that label
		if len(labels) == 1:
			for label, bbox in labels_bboxes:
				logger.debug("only one label {}, adding bboxes {}", label, bboxes)  # pyright: ignore[reportPossiblyUnboundVariable]
				unique_crops.update(bbox)
			continue

		# flatten label_bboxes from tuple[tuple[LABEL, tuple[bbox, ...]]] to tuple[tuple[[LABEL, bbox], ...]]
		# where the first-depth index is the label group corresponding to labels
		labels_bboxes_flattened = flatten_label_bboxes(labels_bboxes)

		# create cartesian combinations for all array lengths n, n-1 ... 1
		labels_bboxes_combinations: list[tuple[LB, ...]] = []

		# this enables rollup
		search_combinations: list[tuple[tuple[LB, ...], ...]] = []
		for combination_constraint in range(len(labels_bboxes_flattened), 0, -1):
			search_combinations.extend(
				combinations(labels_bboxes_flattened, combination_constraint)
			)
			if prefer_largest:  # only look to crop the largest combo(s)
				break

		# the actual cartesian product
		for search_combination in search_combinations:
			labels_bboxes_combinations.extend(product(*search_combination))

		# combine bboxes from combinations
		labels_bboxes_combo: tuple[tuple[LABEL, tuple[int, int, int, int]], ...]
		for labels_bboxes_combo in labels_bboxes_combinations:
			labels, bboxes = zip(*labels_bboxes_combo)
			combined_bbox = combine_bboxes(bboxes)
			if combined_bbox not in unique_crops:
				unique_crops.add((combined_bbox))
				crops.add((labels, combined_bbox))

	return crops


def is_crop_significant(
	ref_img: ImageFile,
	bbox: BBox,
	crop_threshold: float = 2 / 3,
	*,
	crop_regardless=False,
) -> Image.Image | None:
	"""Return crop if crop some ratio smaller than reference image, otherwise None"""
	crop = ref_img.crop(bbox)
	if crop_regardless:
		return crop

	width, height = ref_img.size
	total_pixels_orig = width * height

	crop_width, crop_height = crop.size
	total_pixels_crop = crop_width * crop_height
	if total_pixels_crop > total_pixels_orig * crop_threshold:
		logger.info(
			"skip crop: crop not significant ({:.2f}% > {:.2f}%)",
			100 * total_pixels_crop / total_pixels_orig,
			100 * crop_threshold,
		)
		return None

	return crop


def crop_all(
	im_pth: Path,
	crops: Iterable[list[LABEL]],
	*,
	rollup: bool = False,
	prefer_largest: bool = False,
	crop_regardless: bool = False,
) -> list[Path] | None:
	"""Crop all combinations given a list of things to crop.

	It's way too difficult to dynamically crop, and even if we used AI it isn't
	completely fool proof. AI is used to generate the crops and then the user
	can decide if they want to use said crop or not.
	"""
	if im_pth.suffix not in IMG_EXT:
		return None

	logger.info("starting crop on {}", im_pth.stem)

	ret: list = []

	matching_stem = tuple(CROP.glob(f"{im_pth.stem}*"))
	if matching_stem:  # if can glob then crops already exists
		# logger.info("croped files already exists for {}", im_pth.stem)
		return []

	# # debug show potential crops
	# yres = booru_yolo.detect_with_booru_yolo(im_pth, "yolov8m_as03")
	# plt.imshow(detection_visualize(im_pth, yres))
	# plt.show()
	# # return

	yres = booru_yolo.detect_with_booru_yolo(im_pth, "yolov8m_as03")
	if len(yres) == 0:  # no detections
		logger.info("no crops found")
		return

	logger.debug("all found bboxes: {}", yres)

	threshold = 0.0
	labels_bbox: set[tuple[tuple[LABEL, ...], BBox]] = generate_cartesian_crops(
		yres,
		threshold,
		labels_to_crop=crops,
		# rollup=rollup,
		prefer_largest=prefer_largest,
	)

	logger.info("found {} crops", len(labels_bbox))

	# no bounding boxes found
	if not labels_bbox:
		return

	logger.debug("finalized bboxes for image {}", labels_bbox)

	im: ImageFile = Image.open(im_pth)

	for labels, bbox in labels_bbox:
		crop = is_crop_significant(im, bbox, crop_regardless=crop_regardless)
		if not crop:
			continue

		cropped_out = CROP / (
			f"{im_pth.stem}-crop_" + "_".join(label.value for label in labels) + ".webp"
		)
		cropped_out = exists_handler(cropped_out)
		logger.debug("saving crop to: {}", cropped_out)

		crop.save(
			cropped_out,
			lossless=True,
			method=0,
			quality=0,
		)
		ret.append(cropped_out)

	return ret


def crop_person(
	im_pth: Path, threshold: float = 0.3, *, crop_regardless: bool = False
) -> tuple[BBox, ...] | None:
	if im_pth.suffix not in IMG_EXT:
		return None

	im = Image.open(im_pth)

	pres = person.detect_person(im, level="x", version="v0", conf_threshold=threshold)
	bboxes = tuple(bbox for bbox, label, conf in pres)
	logger.debug("found bboxes above threshold = {}", bboxes)

	# # debug show potential crops
	# plt.imshow(detection_visualize(im_pth, pres))
	# plt.show()
	# return

	out = CROP / f"{im_pth.stem}-crop_person.webp"
	for bbox in bboxes:
		crop = is_crop_significant(im, bbox, crop_regardless=crop_regardless)
		if not crop:
			continue

		out = exists_handler(out)
		crop.save(
			exists_handler(out),
			lossless=True,
			method=0,
			quality=0,
		)

	return bboxes


def tag_img(
	im_pth: Path,
	tokenizer,
	chara_tag_db: Iterable[str],
	delim: str = ",",
	*,
	drop_overlap=False,
	clip_pad_tolerance=65,
	whitelist=[],
	blacklist=[],
	clip_resolve: CLIP_RESOLVE = CLIP_RESOLVE.SMART,
	threshold: float = 0.35,
) -> Path | None:
	"""Tag image and return the written text file

	Takes a list of tags that are considered characters. Used to transfer
	character tags from a reference image to a cropped image.

	Will resolve excessive clip padding. If the intention is to drop (through
	`smart` or `drop`), if we are within the first clip window, no changes will
	be made.

	Cropped images require special handling.
	We cannot trivially use all the tags because cropping will drop some tags
	(since we are literally dropping parts of the image). But we can use it to
	hint on inference what was in the original reference image. However, we
	need to bias towards the inference. This is opposite to how we normally
	handle non-cropped images: bias is towards the user's tags.

	Cropped images must be put through the interrogator.
	"""
	if im_pth.suffix not in IMG_EXT:
		return None

	whitelist = [] if whitelist is None else whitelist
	blacklist = [] if blacklist is None else blacklist
	ref_tags = []
	out = Path(f"./{im_pth.parent}/{im_pth.stem}.txt")

	crop = None
	if "-crop_" in im_pth.stem:
		ref_file, crop = im_pth.stem.split("-crop_")
	else:
		ref_file = im_pth.stem
	existing_reference_tags_pth = DONE / f"{ref_file}.txt"

	def calc_total_tokens(tags: Iterable[str]) -> int:
		tokens = tokenizer.encode(delim.join(tags), return_tensors="pt")
		return tokens.shape[1]

	def get_clip_tokens(tags: Iterable[str]) -> int:
		"""Calculate clip tokens within its window

		Use calc_total_tokens to get total tokens across entire clip windows.
		"""
		token_count = calc_total_tokens(tags)
		return token_count % 75

	def get_clip_padding(tags: Iterable[str]) -> int:
		"""Calculate clip padding within its window

		Use calc_total_tokens to get total tokens across entire clip windows.
		"""
		return 75 - get_clip_tokens(tags)

	def is_excessive_clip_padding(tags: Iterable[str]) -> bool:
		return get_clip_padding(tags) > clip_pad_tolerance

	def resolve_smart(tags: Iterable[str]) -> int:
		"""Drop (-1) or add (+1) tags based on the closest clip window"""
		token_count = calc_total_tokens(tags)
		dist_tolerance = get_clip_padding(tags) - clip_pad_tolerance
		dist_prev_clip_window = get_clip_tokens(tags)
		if dist_tolerance < dist_prev_clip_window or token_count < 75:
			return 1
		else:
			return -1

	if im_pth.suffix not in IMG_EXT:
		return

	logger.info("tag image {}", im_pth.stem)

	if existing_reference_tags_pth.exists():
		ref_tags = get_tags(existing_reference_tags_pth, delim)

	logger.debug("start tags {}", ref_tags)
	# we don't differentiate dropped and ref tags, but oh wells should be fine
	if ref_tags:
		old_tags = set(ref_tags)
		ref_tags = overlap.drop_overlap_tags(ref_tags)
		diff = old_tags.difference(ref_tags)
		if diff:
			logger.debug("drop overlapping: {}", ", ".join(diff))

		logger.debug("after drop overlap {}", ref_tags)

		if not crop:
			if clip_resolve == CLIP_RESOLVE.DROP:
				if calc_total_tokens(ref_tags) < 75:
					write_tags(out, ref_tags)
					return out

			elif clip_resolve == CLIP_RESOLVE.SMART:
				offset_tokens_towards = resolve_smart(ref_tags)
				if offset_tokens_towards == -1:  # would be dropping tags
					fixed_tags = ref_tags[:]
					# handle blacklist
					for tag in ref_tags:
						if tag in blacklist:
							fixed_tags.remove(tag)
					write_tags(out, fixed_tags)
					return out

	minimum_threshold = 0.01
	ratings, general_tags, character_tags = wd14.get_wd14_tags(
		im_pth,
		"EVA02_Large",
		no_underline=True,
		drop_overlap=False,  # will drop later
		general_threshold=minimum_threshold,  # very low for adding if needed
	)

	# when user tags exist, penalize all general tag confs to give room for
	# user+ai bonus
	#
	# in other words, this acts as the upper clamp for only ai-inferenced tags
	#
	# another way to see this is that user+ai tagged tags get a
	# 1-conf_rescale_factor bonus and can fully reach a conf of 1.0
	if ref_tags:
		conf_rescale_factor = 0.66
		for tag, conf in general_tags.items():
			general_tags[tag] = conf * conf_rescale_factor

	# logger.debug(
	# 	"inferenced tags: {}", sorted(general_tags.items(), key=lambda item: -item[1])
	# )

	# prefer user-defined tags, but on crop prefer ai tags
	rating = None
	ref_chara: list[str] = []
	newly_added_tags = []
	for tag in ref_tags:
		if tag in ratings.keys():
			if rating is not None:
				msg = f"Two ratings were defined! There should only be one. ({tag}, {rating[0]})"
				raise ValueError(msg)
			rating = tag

		elif tag in chara_tag_db:
			ref_chara.append(tag)

		else:  # tag is general tag, even if not inferenced
			only_user_tagged = tag not in general_tags.keys()
			if only_user_tagged:
				newly_added_tags.append(tag)
				# on crops, do not include user-only tags within threshold
				# because crops may not include them, but still keep them
				# with the possibility of adding them when we need more tags
				#
				# however, since we fetch initial tags with very low
				# threshold, if the ai hasn't seen it at this point, it's
				# most likely not even in the image
				if crop:
					general_tags[tag] = minimum_threshold
				else:
					user_conf = 0.7
					general_tags[tag] = threshold + (1.0 - threshold) * user_conf
				continue

			# both user and ai tagged
			# IMPORTANT: collection of tags are based on minimum threshold
			general_tags[tag] = max(
				minimum_threshold,
				min(
					# undo rescale
					general_tags[tag] / conf_rescale_factor,  # pyright: ignore[reportPossiblyUnboundVariable]
					1.0,
				),
			)

			# del general_tags[tag]  # I AM HERE FOR TESTING PURPOSES

	if ref_tags:
		logger.debug(
			"general tags after weight adjustments: {}",
			sorted(general_tags.items(), key=lambda item: -item[1]),
		)

	# redrop any overlapping tags
	before_drop = set(order.sort_tags(general_tags, mode="score"))
	general_tags = overlap.drop_overlap_tags(
		general_tags
	)  # return type Mapping denotes read-only, but w/e it returns a dict
	overlapping_tags = before_drop.difference(general_tags.keys())  # pyright: ignore[reportAttributeAccessIssue] # returns Mapping, not List

	filtered_newly_added_tags = []
	for tag in newly_added_tags:
		if (
			not crop
			and tag not in overlapping_tags
			and tag not in chara_tag_db
			and tag not in whitelist
		):
			filtered_newly_added_tags.append(tag)
	if filtered_newly_added_tags:
		logger.warning(
			'tags "{}" were not discovered on inference, misspelling or custom tag?',
			", ".join(filtered_newly_added_tags),
		)

	for tag in dict(general_tags):  # pyright: ignore
		if tag == "belly":
			logger.warning(">>>>>>>>>>>>>>>>>> FOUND BELLY")
		if tag in whitelist:
			general_tags[tag] = 1.0  # pyright: ignore
		elif tag in blacklist + ALWAYS_BLACKLIST:
			del general_tags[tag]  # pyright: ignore
			if tag == "belly":
				logger.warning(">>>>>>>>>>>>>>>>>> DELETE BELLY")
			logger.warning(general_tags)

	resolved_rating = [rating if rating else sort_tags(ratings)[0]]
	resolved_chara_tags = ref_chara if ref_chara else list(character_tags.keys())
	moving_threshold = cull_index(general_tags, threshold=threshold)  # pyright: ignore[reportArgumentType]

	resolved_tags = (
		resolved_rating
		+ resolved_chara_tags
		+ sort_tags(general_tags)[:moving_threshold]
	)

	delta_tags = []

	if clip_resolve == CLIP_RESOLVE.SMART:
		offset_tokens_towards = resolve_smart(resolved_tags)
	# if token_count < 75, we should have written tags and returned
	# in other words, we should never drop tags when within first clip window
	elif clip_resolve == CLIP_RESOLVE.DROP:
		offset_tokens_towards = -1
	elif clip_resolve == CLIP_RESOLVE.ADD:
		offset_tokens_towards = 1

	if is_excessive_clip_padding(resolved_tags):
		logger.debug(
			"clip has excessive clip padding, offset tag window by {}",
			offset_tokens_towards,
		)

	general_tags_sorted = sort_tags(general_tags)
	clip_resolved_tags = resolved_tags[:]
	while is_excessive_clip_padding(clip_resolved_tags):
		delta_tag = general_tags_sorted[
			moving_threshold + 1 if moving_threshold == 1 else moving_threshold
		]
		logger.debug(
			"{} tag {}", "add" if offset_tokens_towards > 0 else "drop", delta_tag
		)
		delta_tags.append(delta_tag)

		if clip_resolve == CLIP_RESOLVE.SMART:
			offset_tokens_towards = resolve_smart(clip_resolved_tags)

		moving_threshold += offset_tokens_towards

		clip_resolved_tags = (
			resolved_rating
			+ resolved_chara_tags
			+ sort_tags(general_tags)[:moving_threshold]
		)
		# this shouldn't be needed anymore as it's handled earlier
		# resolve smart also always returns +1 on clip < 75
		# if offset_tokens_towards == -1 and calc_total_tokens(tags) < 75:
		# 	break
	if delta_tags:
		logger.debug(
			"{}: {}",
			"add" if offset_tokens_towards > 0 else "drop",
			", ".join(delta_tags),
		)

	logger.debug("final tags: {}", clip_resolved_tags)
	write_tags(out, clip_resolved_tags)
	return out

	# out = Path(f"./{im_pth.parent}/{im_pth.stem}.txt")
	# if out.exists():
	# 	tags = ref_captions
	#
	# 	# logger.debug("tags {}", tags)
	# 	# tokens = tokenizer.encode(delim.join(tags), return_tensors="pt")
	# 	# token_count = tokens.shape[1]
	# 	# logger.debug(token_count)
	# 	# if token_count > 75 and token_count % 75 < clip_overflow_tolerance:
	# 	# 	logger.warning(
	# 	# 		"Token count below tolerance {} < {}!",
	# 	# 		token_count % 75,
	# 	# 		clip_overflow_tolerance,
	# 	# 	)
	# 	#
	# 	# 	logger.info("Dropping overlapping tags to prevent excessive padding")
	# 	# 	old_tags = set(tags)
	# 	# 	tags = overlap.drop_overlap_tags(tags)
	# 	# 	logger.info("Dropped tags: {}", old_tags.difference(tags))
	#
	# 	# for now we always drop overlap
	# 	old_tags = set(tags)
	# 	tags = overlap.drop_overlap_tags(tags)
	# 	logger.info("Dropped tags: {}", old_tags.difference(tags))
	#
	# 	tokens = tokenizer.encode(delim.join(tags), return_tensors="pt")
	# 	token_count = tokens.shape[1]
	# 	# logger.debug(token_count)
	#
	# 	# tokens still too high, order tags and drop the least confidence,
	# 	# weight adjusted, we drop after this if block
	# 	if token_count > 75 and token_count % 75 < clip_overflow_tolerance:
	# 		logger.debug("still too high, sorting tags")
	# 		ratings, general_tags, character_tags = wd14.get_wd14_tags(
	# 			im_pth,
	# 			"EVA02_Large",
	# 			no_underline=True,
	# 			drop_overlap=True,
	# 		)
	#
	# 		# grab rating if exists
	# 		rating = None
	# 		for tag in tags[:]:
	# 			if tag in ratings.keys():
	# 				rating = tag
	#
	# 		resolved_rating = (
	# 			rating
	# 			if rating
	# 			else sorted(ratings.items(), key=lambda item: item[1], reverse=True)[0][
	# 				0
	# 			]
	# 		)
	#
	# 		# If reference captions exist, prefer to any existing character tags
	# 		# there, otherwise use what we inference
	# 		ref_chara: [str] = []
	# 		for tag in tag:
	# 			if tag in chara_tag_db:
	# 				ref_chara.append(tag)
	#
	# 		resolved_chara_tags = (
	# 			ref_chara if ref_chara else list(character_tags.keys())
	# 		)
	#
	# 		# bias towards user-defined tags
	# 		# if interrogator didn't find it, guess it goes somewhere in the middle
	# 		for tag in tags:  # not ref_captions, we want already dropped
	# 			if tag not in chara_tag_db and tag not in ratings:
	# 				if tag not in general_tags:
	# 					general_tags[tag] = 0.8
	# 				else:
	# 					general_tags[tag] = min(1.0, general_tags[tag] * 1.2)
	#
	# 		sorted_general_tags = [
	# 			tag
	# 			for tag, confidence in sorted(
	# 				general_tags.items(), key=lambda item: -item[1]
	# 			)
	# 		]
	#
	# 		tags = [resolved_rating] + resolved_chara_tags + sorted_general_tags
	#
	# 	# token count still too high
	# 	tokens = tokenizer.encode(delim.join(tags), return_tensors="pt")
	# 	# logger.debug(token_count)
	# 	token_count = tokens.shape[1]
	# 	if token_count > 75 and token_count % 75 < clip_overflow_tolerance:
	# 		logger.warning(
	# 			"Token count below tolerance {} < {}!",
	# 			token_count % 75,
	# 			clip_overflow_tolerance,
	# 		)
	#
	# 		logger.info(
	# 			"Manually dropping tags until within clip overflow tolerance..."
	# 		)
	#
	# 		while token_count > 75 and token_count % 75 < clip_overflow_tolerance:
	# 			truncated_tag = tags.pop(-1)
	# 			logger.info("Drop: {}", truncated_tag)
	#
	# 			tokens = tokenizer.encode(delim.join(tags), return_tensors="pt")
	# 			token_count = tokens.shape[1]
	#
	# else:  # no existing tags exist
	# 	ratings, general_tags, character_tags = wd14.get_wd14_tags(
	# 		im_pth,
	# 		"EVA02_Large",
	# 		no_underline=True,
	# 		drop_overlap=True,  # for now, always drop?
	# 		# general_mcut_enabled=True,
	# 		# character_mcut_enabled=True,
	# 		#
	# 		# Mcut dynamically determines threshold as the point of max
	# 		# difference. Might be useful for training concepts...? But not style
	# 		# as it prunes too much.
	# 	)
	#
	# 	rating = sorted(ratings.items(), key=lambda item: item[1], reverse=True)[0][0]
	#
	# 	# If reference captions exist, prefer to any existing character tags
	# 	# there, otherwise use what we inference
	# 	if ref_captions:
	# 		ref_chara: [str] = []
	# 		for tag in ref_captions:
	# 			if tag in chara_tag_db:
	# 				ref_chara.append(tag)
	#
	# 	resolved_chara_tags = (
	# 		ref_chara if ref_chara else list(character_tags.keys())
	# 	)
	#
	# 	sorted_general_tags = [
	# 		tag
	# 		for tag, confidence in sorted(
	# 			general_tags.items(), key=lambda item: -item[1]
	# 		)
	# 	]
	# 	tags = [rating] + resolved_chara_tags + sorted_general_tags
	# 	# logger.info("Raw, original tags: {}", tags)
	#
	# 	# # for now we always drop
	# 	# tokens = tokenizer.encode(delim.join(tags), return_tensors="pt")
	# 	# token_count = tokens.shape[1]
	# 	# # try reducing tokens
	# 	# if token_count > 75 and token_count % 75 < clip_overflow_tolerance:
	# 	# 	logger.warning(
	# 	# 		"Token count below tolerance {} < {}!",
	# 	# 		token_count % 75,
	# 	# 		clip_overflow_tolerance,
	# 	# 	)
	# 	# 	tags = overlap.drop_overlap_tags(tags)
	# 	#
	# 	# 	logger.info("Dropping overlapping tags to prevent excessive padding")
	#
	# 	# token count still too high
	# 	tokens = tokenizer.encode(delim.join(tags), return_tensors="pt")
	# 	token_count = tokens.shape[1]
	# 	if token_count > 75 and token_count % 75 < clip_overflow_tolerance:
	# 		logger.warning(
	# 			"Token count below tolerance {} < {}!",
	# 			token_count % 75,
	# 			clip_overflow_tolerance,
	# 		)
	#
	# 		logger.info(
	# 			"Manually dropping tags until within clip overflow tolerance..."
	# 		)
	#
	# 		while token_count > 75 and token_count % 75 < clip_overflow_tolerance:
	# 			truncated_tag = tags.pop(-1)
	# 			logger.info("Drop: {}", truncated_tag)
	#
	# 			tokens = tokenizer.encode(delim.join(tags), return_tensors="pt")
	# 			token_count = tokens.shape[1]
	#
	# logger.debug("token count: {}", token_count)
	# logger.debug("final tags: {}", delim.join(tags))
	#
	# write_tags(out, tags)
	# return out


def cull_threshold(
	tags: Mapping[str, float], *, threshold: float = 0.35
) -> Mapping[str, float]:
	culled_tags = {}
	for tag, conf in tags.items():
		if conf > threshold:
			culled_tags[tag] = conf
	return culled_tags


def cull_index(tags: Mapping[str, float], *, threshold: float = 0.35) -> int:
	for index, conf in enumerate(sorted(tags.values(), reverse=True)):
		if conf < threshold:
			return index

	msg = f"Threshold {threshold} is beyond potential confidecnes"
	raise IndexError(msg)


def upscale(im_pth: Path, total_pixels: int = 768 * 768) -> Path | None:
	"""Upscale an image to total pixels

	Usually you want to upscale to your lowest reasonable bucket (512*512).
	"""
	if im_pth.suffix not in IMG_EXT:
		return None

	out = UP / f"{im_pth.stem}.webp"
	if out.exists():
		out.touch(exist_ok=True)
		logger.info("upscaling already exists for {}", im_pth.stem)
		return

	im: ImageFile = Image.open(im_pth)
	px = im.width * im.height
	if px < total_pixels:  # x2 if < 0.5MP
		im_up = upscale_with_cdc(im_pth, "HGSR-MHR_X2_1680")
		im_up.save(
			out,
			lossless=True,
			method=0,
			quality=0,
		)

	return im_pth


def add_suffix_to_file(im_pth: Path, suffix: str) -> Path:
	return im_pth.parent / f"{im_pth.stem}{suffix}{im_pth.suffix}"


def restore_scu(im_pth: Path) -> Path | None:
	if im_pth.suffix not in IMG_EXT:
		return None

	out = SCU / f"{im_pth.stem}.webp"
	if out.exists():
		out.touch(exist_ok=True)
		logger.info("scu restore already exists for {}", im_pth.stem)
		return None

	logger.info("restoring image {}", im_pth.stem)
	im: ImageFile = Image.open(im_pth)
	scu: Image.Image = scunet.restore_with_scunet(
		im, tile_size=256, tile_overlap=32, batch_size=4
	)
	scu.save(
		out,
		lossless=True,
		method=0,
		quality=0,
	)


def get_tags(pth: Path, delim=",") -> list[str]:
	with pth.open("r") as fd:
		contents = fd.read().strip()

	return contents.split(delim)


def write_tags(pth: Path, tags: Iterable[str], delim=","):
	with pth.open("w") as fd:
		fd.write(delim.join(tags))


def prepend_tag(pth: Path, tag: str, delim=","):
	if pth.suffix != ".txt":
		return

	tags = get_tags(pth, delim)
	if tag in tags:
		tags.remove(tag)

	write_tags(pth, [tag] + tags, delim)


def downscale(im_pth: Path, total_pixels: int = 2024 * 2024) -> Path | None:
	"""Downscale image so total images are not beyond total pixels.

	Usually you want to downscale to your highest bucket (2024*2024).
	"""
	if im_pth.suffix not in IMG_EXT:
		return None

	out = DOWNSCALE / f"{im_pth.stem}.webp"
	if out.exists():
		out.touch(exist_ok=True)
		logger.info("Downscaled version already eixists for {}", out.stem)
		return None

	img = Image.open(im_pth)
	width, height = img.size

	if width * height >= total_pixels:
		logger.info("downscaling {}", im_pth.stem)
		aspect_ratio = width / height
		new_width = int((total_pixels * aspect_ratio) ** 0.5)
		new_height = int(new_width / aspect_ratio)
		img_resized = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
	else:
		logger.info("image is smaller than desired downscale {}", im_pth.stem)
		# just save as webp for compression and easier overwrite management on move
		img_resized = img

	img_resized.save(
		out,
		lossless=True,
		method=0,
		quality=0,
	)

	return out


def main():
	parser = argparse.ArgumentParser(description="Preprocess images for AI training.")

	parser.add_argument("--clean", action="store_true", help="clean workspace")

	# TODO: let user decide what to keep
	# let user choose how to crop
	# comma separated for distinct crops
	# combine crop bboxes with "+" to combine to single crop
	# ---
	# at the moment, only keeps portrait/upper body/cowboy shot, trashes head
	# if keep all, keeps both

	parser.add_argument(
		"--crop",
		# default=["head"],
		action="extend",
		nargs="*",
		help="do specified crops (default %(default)s)",
	)
	parser.add_argument(
		"--crop-person",
		# default=["head"],
		action="store_true",
		help="do crop person",
	)
	parser.add_argument(
		"--crop-rollup",
		action="store_true",
		help="rollup crops (e.g. head+bust+belly -> head+bust+belly,head+bust,head)",
	)
	parser.add_argument(
		"--crop-regardless",
		action="store_true",
		help="crop images even if insigificant",
	)
	parser.add_argument(
		"--prefer-largest", action="store_true", help="prefer largest crop rollup"
	)
	parser.add_argument("--downscale", action="store_true", help="downscale images")
	parser.add_argument("--upscale", action="store_true", help="upscale images")
	parser.add_argument("--restore", action="store_true", help="restore images")
	parser.add_argument(
		"--move",
		action="store_true",
		help="move finalized images and captions to ./done/",
	)
	parser.add_argument("--tag", action="store_true", help="tag images in ./done/")
	parser.add_argument("--tag-prepend", help="prepend tag to all captions in ./done/")
	parser.add_argument(
		"--tag-whitelist",
		# default=["head"],
		action="extend",
		nargs="*",
		help="whitelist tag on tagging",
	)
	parser.add_argument(
		"--tag-blacklist",
		# default=["head"],
		action="extend",
		nargs="*",
		help="blacklist tag on tagging",
	)
	parser.add_argument(
		"--resolve",
		type=CLIP_RESOLVE,
		choices=list(CLIP_RESOLVE),
		help="How to resolve excessive clip padding",
	)
	parser.add_argument(
		"--clip-padding-tolerance",
		type=int,
		default=65,
		help="Maximum allowed clip padding.",
	)
	parser.add_argument(
		"--threshold",
		type=float,
		default="0.35",
		help="Initial cutoff for tags before resolving for clip padding",
	)

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

	# # Sandbox
	# _, _, characters = load_danbooru_tags()
	# logger.info(characters)
	# exit()
	# # Sandbox END

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

	if args.crop or args.stage_1:
		logger.info("start crop")
		# logger.debug(args.crop)
		crops = parse_crops(args.crop)
		# logger.debug("{}", crops)
		for pth in RAW.iterdir():
			crop_all(
				pth,
				crops,
				rollup=args.crop_rollup,
				prefer_largest=args.prefer_largest,
				crop_regardless=args.crop_regardless,
			)
			# crop_dynamic(
			# 	pth, crops, rollup=args.crop_rollup, prefer_largest=args.prefer_largest
			# )

	if args.crop_person:
		for pth in RAW.iterdir():
			crop_person(pth, crop_regardless=args.crop_regardless)

	if args.crop or args.crop_person or args.stage_1:
		logger.success("done crop")
		logger.warning(
			"You should manually verify the crops before moving onto --stage-2."
		)

	if args.downscale or args.stage_1:
		logger.info("start downscale")
		for pth in RAW.iterdir():
			downscale(pth)
		for pth in CROP.iterdir():
			downscale(pth)
		logger.success("done downscale")

	if args.restore or args.stage_1:
		logger.info("start restore")
		for file in SCUTODO.iterdir():
			os.remove(file)

		for to_file in RAW.iterdir():
			# currently only handles unix path, sry windows lol get wrecked
			if to_file.suffix == ".txt" and not to_file.name.startswith("."):
				continue
			from_file = SCUTODO / to_file.name
			os.symlink(to_file.absolute(), from_file)

		for to_file in CROP.iterdir():
			from_file = SCUTODO / to_file.name
			os.symlink(to_file.absolute(), from_file)

		for to_file in DOWNSCALE.iterdir():
			from_file = SCUTODO / to_file.name
			if from_file.exists():  # overwrite raw/cropped with downscaled
				os.remove(from_file)
			os.symlink(to_file.absolute(), from_file)

		for pth in SCUTODO.iterdir():
			restore_scu(pth)
		logger.success("done restore")

	if args.upscale or args.stage_2:
		logger.info("start upscale")
		for pth in SCU.iterdir():
			upscale(pth)
		logger.success("done upscale")

	if args.move or args.stage_2:
		logger.info("start move")
		for pth in SCU.iterdir():
			shutil.copy(pth, DONE / pth.name)
		# move captions if already exists
		for pth in RAW.iterdir():
			if pth.suffix == ".txt":
				shutil.copy(pth, DONE / pth.name)

		logger.success("done move images")

	if args.tag:
		logger.info("start tag images")
		_, _, characters = load_danbooru_tags()

		tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(
			"./clip-vit-base-patch32/"
		)

		for pth in DONE.iterdir():
			tag_img(
				pth,
				tokenizer,
				characters,
				drop_overlap=True,
				clip_pad_tolerance=args.clip_padding_tolerance,
				whitelist=args.tag_whitelist,
				blacklist=args.tag_blacklist,
				clip_resolve=args.resolve,
				threshold=args.threshold,
			)
		logger.success("done tag images")

	if args.tag_prepend:
		logger.info("start tag prepend")
		for pth in DONE.iterdir():
			prepend_tag(pth, args.tag_prepend)
		logger.success("done tag prepend")

	logger.success("done")


if __name__ == "__main__":
	main()
