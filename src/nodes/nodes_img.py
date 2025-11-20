# -*- coding: utf-8 -*-
# Copyright (c) 2025 Salvador E. Tropea
# Copyright (c) 2025 Instituto Nacional de Tecnologïa Industrial
# License: GPL-3.0
#
# Project: ComfyUI-ImageMisc
# Credits:
# - ImagePad, ImageResize and ResizeMask are from Kijai (https://github.com/kijai/ComfyUI-KJNodes/) v1.1.7
# - Assisted by Gemini 2.5 Pro
from collections import defaultdict
from copy import deepcopy
import csv
import numpy as np
import os
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont  # Import the Python Imaging Library
import random
import re
from seconohe.apply_mask import apply_mask
from seconohe.color import color_to_rgb_float, color_to_rgb_uint8
from seconohe.comfy_misc import (get_input_directory, get_output_directory, MAX_RESOLUTION, PromptServer, IO, ComfyNodeABC,
                                 ExecutionBlocker, upscale_methods)
from seconohe.downloader import download_file
from seconohe.foreground_estimation.affce import affce
from seconohe.foreground_estimation.fmlfe import fmlfe, IMPL_PRIORITY
from seconohe.tensor import batched_min_max_norm
from seconohe.torch import get_default_comfy_device, get_canonical_device
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from typing import Optional

# We are the main source, so we use the main_logger
from . import main_logger, F_POINTS
from .helpers import load_image_wrapper, load_images_wrapper, save_image, upscale, upscale_comfy
from .s_measure import get_s_measure
from .e_measure import get_e_measure
from .f_measure import get_f_measure, get_weighted_f_measure


logger = main_logger
BASE_CATEGORY = "image"
IO_CATEGORY = "io"
MANIPULATION_CATEGORY = "manipulation"
NORMALIZATION = "normalization"
VALIDATION = "validation"
FOREGROUND = "foreground"
BLUR_SIZE_OPT = (IO.INT, {"default": 90, "min": 1, "max": 255, "step": 1, })
BLUR_SIZE_TWO_OPT = (IO.INT, {"default": 6, "min": 1, "max": 255, "step": 1, })
COLOR_OPT = (IO.STRING, {
                "default": "#000000",
                "tooltip": "Color for fill.\n"
                           "Can be an hexadecimal (#RRGGBB).\n"
                           "Can comma separated RGB values in [0-255] or [0-1.0] range."})
DEFAULT_UPSCALE = 'bicubic'     # transforms.InterpolationMode.BICUBIC.value
MASK_UPSCALE = 'nearest-exact'  # transforms.InterpolationMode.NEAREST_EXACT.value
BEST_UPSCALE = 'lanczos'        # transforms.InterpolationMode.LANCZOS.value
UPSCALE_OPT = (upscale_methods, {  # [mode.value for mode in transforms.InterpolationMode]
                "default": DEFAULT_UPSCALE,
                "tooltip": "Interpolation method for image resize"
                })
UPSCALE_OPT_MASK = deepcopy(UPSCALE_OPT)
UPSCALE_OPT_MASK[1]["default"] = MASK_UPSCALE
PAD_SIZE_OPT = (IO.INT, {"default": 0, "min": 0, "max": MAX_RESOLUTION, "step": 1, })
SIZE_OPT = (IO.INT, {"default": 512, "min": 0, "max": MAX_RESOLUTION, "step": 1})
SIZE_OPT_FI = deepcopy(SIZE_OPT)
SIZE_OPT_FI[1]["forceInput"] = True
SIZE_OPT_FI[1]["tooltip"] = ("Connect both `target` inputs\n"
                             "If 0 the size of the image is used\n"
                             "Overrides left/right/top/bottom")
SIZE_OPT[1]["tooltip"] = "Used when no `get_image_size` is provided"
PAD_TRANS = (IO.FLOAT, {
                "default": 1.0,
                "min": 0.0,
                "max": 1.0,
                "step": 0.1,
                "display": "number",
                "tooltip": ("The transparency for the padded area for all modes except `edge_pixel`."
                            "1.0 is fully transparent, 0.0 is fully opaque.")})
NORM_PARAM = (IO.FLOAT, {
                "default": 1.0,
                "min": 0.0,
                "max": 1.0,
                "step": 0.1,
                "display": "number"})
MAX_FILES = 0xffffffffffffffff
EMBED_TRANSPARENCY = (IO.BOOLEAN, {
                        "default": False,
                        "tooltip": "Create RGBA images when they have transparency."})
SAVE_PROMPT = (IO.BOOLEAN, {
                "default": False,
                "tooltip": "Save prompt submitted to ComfyUI"})
SAVE_WORKFLOW = (IO.BOOLEAN, {
                  "default": False,
                  "tooltip": "Save the ComfyUI workflow"})
SHOW_PREVIEW = (IO.BOOLEAN, {
                 "default": True,
                 "tooltip": "Show a preview of the images"})
SOD_NAMES = {'mae': "MAE", 'max_f_mes': "Max F-measure", 'adp_f_mes': "Adp F-measure",
             'max_e_mes': "Max E-measure", 'e_mes': "E-measure mean", 'adp_e_mes': "Adp E-measure",
             's_mes': "S-measure", 'wf_mes': "Weighted F-measure"}
REVERSE_SOD_NAMES = {v: k for k, v in SOD_NAMES.items()}
IS_VECT_NAME = re.compile(r"(F|FP|FR|E)\((\d+)\)")
VECT_NAMES = {'e', 'f', 'fp', 'fr'}
# A dictionary to cache loaded fonts
font_cache = {}


def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    """Converts a single image tensor (H, W, C) [0, 1] to a Pillow Image."""
    np_image = (tensor.cpu().numpy() * 255).astype(np.uint8)

    if tensor.dim() == 2:
        return Image.fromarray(np_image, 'L')
    else:
        return Image.fromarray(np_image, 'RGB')


def pil_to_tensor(pil_image: Image.Image) -> torch.Tensor:
    """Converts a Pillow Image to a tensor (H, W, C) [0, 1]."""
    np_image = np.array(pil_image).astype(np.float32) / 255.0
    return torch.from_numpy(np_image)


def parse_size(size_str, reference_dim):
    """Parses a size string which can be pixels or a percentage."""
    size_str = size_str.strip()
    if size_str.endswith('%'):
        try:
            percentage = float(size_str[:-1])
            return int(reference_dim * (percentage / 100.0))
        except ValueError:
            return 0
    else:
        try:
            return int(size_str)
        except ValueError:
            return 0


def send_progress_text(unique_id, msg):
    if unique_id and PromptServer is not None:
        try:
            PromptServer.instance.send_progress_text(msg, unique_id)
        except Exception:
            pass
    else:
        logger.info(msg)


def expand_header_keys(header):
    """ Convert a compact list of keys into a an expanded for a CSV """
    h = []
    for key in header:
        if key in VECT_NAMES:
            key = key.upper()
            h.extend([f"{key}({c})" for c in range(F_POINTS)])
        else:
            h.append(SOD_NAMES[key])
    return h


def expand_header(header):
    """ Convert a compact list of keys into a an expanded for a CSV """
    h = []
    for key in header:
        if key.lower() in VECT_NAMES:
            h.extend([f"{key}({c})" for c in range(F_POINTS)])
        else:
            h.append(key)
    return h


def expand_data_row(metric_dict, metric_keys_ordered, formated=False):
    """ Expand a data row (containing floats and tensors) to a CSV row """
    row_data = []
    for key in metric_keys_ordered:
        if key in VECT_NAMES:
            data = metric_dict.get(key)
            if formated:
                row_data.extend([f"{data[c].item():.4f}" for c in range(F_POINTS)])
            else:
                row_data.extend([str(data[c].item()) for c in range(F_POINTS)])
        else:
            if formated:
                row_data.append(f"{metric_dict.get(key, 0.0):.4f}")
            else:
                row_data.append(str(metric_dict.get(key, "")))
    return row_data


# Define sort methods for the node input
sort_methods = [
    "None",
    "Alphabetical (ASC)",
    "Alphabetical (DESC)",
    "Numerical (ASC)",
    "Numerical (DESC)",
    "Datetime (ASC)",
    "Datetime (DESC)",
    "Random",
]


# Helper function to extract the first number from a string for sorting
def extract_first_number(s):
    match = re.search(r'\d+', s)
    return int(match.group()) if match else float('inf')


# Sorting function to be used on the lists
def sort_by(items, base_path='.', method=None, random_seed=1):
    def fullpath(x): return os.path.join(base_path, x)

    def get_timestamp(path):
        try:
            return os.path.getmtime(path)
        except FileNotFoundError:
            return float('-inf')

    if method == "Alphabetical (ASC)":
        return sorted(items)
    elif method == "Alphabetical (DESC)":
        return sorted(items, reverse=True)
    elif method == "Numerical (ASC)":
        return sorted(items, key=lambda x: extract_first_number(os.path.splitext(x)[0]))
    elif method == "Numerical (DESC)":
        return sorted(items, key=lambda x: extract_first_number(os.path.splitext(x)[0]), reverse=True)
    elif method == "Datetime (ASC)":
        return sorted(items, key=lambda x: get_timestamp(fullpath(x)))
    elif method == "Datetime (DESC)":
        return sorted(items, key=lambda x: get_timestamp(fullpath(x)), reverse=True)
    elif method == "Random":
        random.seed(random_seed)
        random.shuffle(items)
        return items
    else:
        return items


class ImageDownload(ComfyNodeABC):
    """
    Downloads an image to ComfyUI's 'input' directory if it doesn't exist.
    Then loads it using the built-in LoadImage logic.
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "base_url": (IO.STRING, {
                    "default":
                        "https://raw.githubusercontent.com/set-soft/AudioSeparation/refs/heads/main/example_workflows/",
                    "tooltip": "The base URL where the image file is located."
                }),
                "filename": (IO.STRING, {
                    "default": "audioseparation_logo.jpg",
                    "tooltip": "The name of the image file to download (e.g., photo.jpg, art.png)."
                }),
            },
            "optional": {
                "image_bypass": (IO.IMAGE, {
                     "tooltip": "If this image is present will be used instead of the downloaded one"
                }),
                "mask_bypass": (IO.MASK, {"tooltip": "If this mask is present will be used instead of the downloaded one"}),
                "local_name": (IO.STRING, {
                    "default": "",
                    "tooltip": "The name used locally. Leave empty to use `filename`"
                }),
                "embed_transparency": EMBED_TRANSPARENCY,
            }
        }

    RETURN_TYPES = (IO.IMAGE, IO.MASK, IO.STRING)
    RETURN_NAMES = ("image", "alpha_mask", "file_name")
    FUNCTION = "load_or_download_image"
    CATEGORY = BASE_CATEGORY + "/" + IO_CATEGORY
    UNIQUE_NAME = "SET_ImageDownload"
    DISPLAY_NAME = "Image Download and Load"
    # This node stores a result to disk. So this IS an output node.
    # It can be used without connecting any other node.
    # Declaring it as output helps with the preview mechanism.
    OUTPUT_NODE = True

    def load_or_download_image(self, base_url: str, filename: str, image_bypass: Optional[torch.Tensor] = None,
                               mask_bypass: Optional[torch.Tensor] = None, local_name: str = None,
                               embed_transparency: bool = False):
        # If we have something at the bypass inputs use it
        if image_bypass is not None or mask_bypass is not None:
            if image_bypass is None:
                # Just a mask
                assert mask_bypass is not None, "This should not be possible if image_bypass is None"  # For mypy
                image_bypass = torch.zeros(mask_bypass.shape + (3,), dtype=torch.float32, device="cpu")
                logger.warning("ImageDownload: Returning an empty image")
            elif mask_bypass is None:
                # This is ComfyUI behavior when we don't have transparency
                mask_bypass = torch.zeros((64, 64), dtype=torch.float32, device="cpu").unsqueeze(0)
                logger.warning("ImageDownload: Returning an empty mask")
            return (image_bypass, mask_bypass)

        save_dir = get_input_directory()
        dest_fname = local_name or filename
        local_filepath = os.path.join(save_dir, dest_fname)

        if not os.path.exists(local_filepath):
            logger.info(f"File '{filename}' not found locally. Attempting to download.")

            if not base_url.endswith('/'):
                base_url += '/'
            download_url = base_url + filename

            try:
                download_file(logger, url=download_url, save_dir=save_dir, file_name=dest_fname, kind="image")
            except Exception as e:
                logger.error(f"Download failed for {download_url}: {e}", exc_info=True)
                raise
        else:
            logger.info(f"Found existing file, skipping download: '{local_filepath}'")

        return load_image_wrapper(dest_fname, embed_transparency, filename)


class ImageLoad(ComfyNodeABC):
    """ Loads an image from any path """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "file_name": (IO.STRING, {
                    "tooltip": "The file name of the image to load"
                }),
                "batch_size": (IO.INT, {
                    "default": 1,
                    "min": 1,
                    "max": 64,
                    "tooltip": "The number of images to create in the batch"
                }),
            },
            "optional": {
                "embed_transparency": EMBED_TRANSPARENCY,
                "show_preview": SHOW_PREVIEW
            }
        }

    RETURN_TYPES = (IO.IMAGE, IO.MASK, IO.STRING)
    RETURN_NAMES = ("image", "alpha_mask", "file_name")
    OUTPUT_IS_LIST = (True, True, True)
    FUNCTION = "execute"
    CATEGORY = BASE_CATEGORY + "/" + IO_CATEGORY
    UNIQUE_NAME = "SET_ImageLoad"
    DISPLAY_NAME = "Load Image from Path"
    INPUT_IS_LIST = True

    def execute(self, file_name, batch_size, embed_transparency, show_preview):
        # Flatten arguments that aren't really expected to be lists
        batch_size = batch_size[0]
        embed_transparency = embed_transparency[0]
        show_preview = show_preview[0]

        return load_images_wrapper(file_name, embed_transparency, show_preview=show_preview, batch_size=batch_size)


class MaskLoad(ComfyNodeABC):
    """ Loads an image from any path using it as a mask """
    _color_channels = ["red", "green", "blue", "alpha"]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "file_name": (IO.STRING, {
                    "tooltip": "The file name of the image to load"
                }),
                "batch_size": (IO.INT, {
                    "default": 1,
                    "min": 1,
                    "max": 64,
                    "tooltip": "The number of images to create in the batch"
                }),
            },
            "optional": {
                "channel": (cls._color_channels, ),
                "show_preview": SHOW_PREVIEW
            }
        }

    RETURN_TYPES = (IO.MASK, IO.STRING)
    RETURN_NAMES = ("mask", "file_name")
    OUTPUT_IS_LIST = (True, True)
    FUNCTION = "execute"
    CATEGORY = BASE_CATEGORY + "/" + IO_CATEGORY
    UNIQUE_NAME = "SET_MaskLoad"
    DISPLAY_NAME = "Load Mask from Path"
    INPUT_IS_LIST = True

    def execute(self, file_name, batch_size, channel, show_preview):
        # Flatten arguments that aren't really expected to be lists
        batch_size = batch_size[0]
        channel = channel[0]
        show_preview = show_preview[0]

        return load_images_wrapper(file_name, show_preview=show_preview, batch_size=batch_size, channel=channel)


class ImageSave(ComfyNodeABC):
    """ Saves an image to an arbitrary path """
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": (IO.IMAGE, {"tooltip": "The images to save."}),
                "filename": (IO.STRING, {"default": "", "tooltip": "The file name for the image"})
            },
            "optional": {
                "show_preview": SHOW_PREVIEW,
                "save_prompt": SAVE_PROMPT,
                "save_workflow": SAVE_WORKFLOW
            },
            "hidden": {
                "prompt": "PROMPT",
                "extra_pnginfo": "EXTRA_PNGINFO"
            },
        }

    RETURN_TYPES = ()
    FUNCTION = "execute"

    OUTPUT_NODE = True
    INPUT_IS_LIST = True

    CATEGORY = BASE_CATEGORY + "/" + IO_CATEGORY
    UNIQUE_NAME = "SET_ImageSave"
    DISPLAY_NAME = "Save Image to Path"

    def execute(self, image, filename, show_preview, save_prompt, save_workflow, prompt=None,
                extra_pnginfo=None):
        # Flatten arguments that aren't really expected to be lists
        if not save_prompt[0]:
            prompt = None
        elif prompt is not None:
            prompt = prompt[0]
        if not save_workflow[0]:
            extra_pnginfo = None
        elif extra_pnginfo is not None:
            extra_pnginfo = extra_pnginfo[0]
        show_preview = show_preview[0]

        return save_image(image, filename, prompt, extra_pnginfo, show_preview=show_preview)


class MaskSave(ComfyNodeABC):
    """ Saves a mask to an arbitrary path """
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "mask": (IO.MASK, {"tooltip": "The mask to save."}),
                "filename": (IO.STRING, {"default": "", "tooltip": "The file name for the image"})
            },
            "optional": {
                "show_preview": SHOW_PREVIEW,
            },
        }

    RETURN_TYPES = ()
    FUNCTION = "execute"

    OUTPUT_NODE = True
    INPUT_IS_LIST = True

    CATEGORY = BASE_CATEGORY + "/" + IO_CATEGORY
    UNIQUE_NAME = "SET_MaskSave"
    DISPLAY_NAME = "Save Mask to Path"

    def execute(self, mask, filename, show_preview):
        return save_image(mask, filename, show_preview=show_preview[0])


class ImageDataset(ComfyNodeABC):
    """
    A ComfyUI node to prepare lists of images for validation tasks,
    such as Salient Object Detection.
    """
    # Define valid image extensions
    valid_extensions = ['.jpg', '.jpeg', '.png', '.webp']

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "source": (IO.STRING, {
                    "default": "./dataset/im",
                    "tooltip": "Path to the images.\nRelative to ComfyUI input"
                }),
                "pattern": (IO.STRING, {
                    "default": ".*",
                    "tooltip": "Python regex to match source images."
                }),
                "destination": (IO.STRING, {
                    "default": "./result",
                    "tooltip": "Path for the result images.\nRelative to ComfyUI output"
                }),
                "dest_ext": (IO.STRING, {
                    "default": "png",
                    "tooltip": "Extension for the destination images.\nEmpty means same as source"
                }),
            },
            "optional": {
                "reference": (IO.STRING, {
                    "default": "./dataset/gt",
                    "tooltip": "Path for the reference images.\nRelative to ComfyUI input"
                }),
                "sort_method": (sort_methods,),
                "image_load_cap": (IO.INT, {
                    "default": 1,
                    "min": 0,
                    "max": MAX_FILES,
                    "tooltip": "How many files to load at once\n"
                               "0 means infinite\n"
                               "Use 1 and queue N runs for low memory usage"
                }),
                "skip_first_images": (IO.INT, {
                    "default": 0,
                    "min": 0,
                    "max": MAX_FILES,
                    "tooltip": "How many file we will skip before starting to process"
                }),
                "select_every_nth": (IO.INT, {
                    "default": 1,
                    "min": 1,
                    "max": MAX_FILES,
                    "tooltip": "Keeps only the first of every n files and discard the rest"
                }),
                "random_seed": (IO.INT, {
                    "default": 1,
                    "min": 0,
                    "max": MAX_FILES,
                    "tooltip": "Random seed used for the random sorting"
                }),
                "block_when_finished": (IO.BOOLEAN, {
                    "default": True,
                    "tooltip": "Block the execution when all images are processed"
                }),
            }
        }

    RETURN_TYPES = (IO.STRING, IO.STRING, IO.STRING,)
    RETURN_NAMES = ("images", "results", "references",)
    # Tell ComfyUI that the outputs of this node are lists.
    OUTPUT_IS_LIST = (True, True, True)
    FUNCTION = "execute"
    CATEGORY = BASE_CATEGORY + "/" + VALIDATION
    UNIQUE_NAME = "SET_ImageDataset"
    DISPLAY_NAME = "List Images from Dataset"

    @classmethod
    def has_valid_extension(cls, filename):
        return Path(filename).suffix.lower() in cls.valid_extensions

    @classmethod
    def IS_CHANGED(cls, source, pattern, destination, dest_ext, reference=None, sort_method="None",
                   image_load_cap=1, skip_first_images=0, select_every_nth=1, random_seed=1,
                   block_when_finished=False):
        # We always indicate an evaluation is needed.
        # 1) We must check the directory to know it, so we don't have any advantage on doing it now.
        # 2) ComfyUI makes any check impossible, if source is connected to a node it will be None
        return float("NaN")

    def execute(self, source, pattern, destination, dest_ext, reference=None, sort_method="None",
                image_load_cap=1, skip_first_images=0, select_every_nth=1, random_seed=1,
                block_when_finished=False):
        # Here self isn't really needed, our state is the filesystem
        source_dir = Path(get_input_directory(), source)
        dest_dir = Path(get_output_directory(), destination)
        ref_dir = Path(get_input_directory(), reference) if reference else None

        # Ensure directories exist
        source_dir.mkdir(parents=True, exist_ok=True)
        dest_dir.mkdir(parents=True, exist_ok=True)
        if ref_dir:
            ref_dir.mkdir(parents=True, exist_ok=True)

        images = []
        results = []
        references = []

        # Compile the regex pattern
        try:
            compiled_pattern = re.compile(pattern)
        except re.error as e:
            raise ValueError(f"Invalid regex pattern: {e}")

        # Get all files in the source directory
        source_files = [f for f in os.listdir(source_dir) if (source_dir / f).is_file()]
        n_files = len(source_files)
        if not n_files:
            raise ValueError("No files to process")
        logger.debug(f"Found {n_files} files in {source_dir}")

        # Filter the images
        source_files = [f for f in source_files if self.has_valid_extension(f) and compiled_pattern.search(f)]
        n_files = len(source_files)
        if not n_files:
            raise ValueError("No images to process after applying filters")
        logger.debug(f"{n_files} images after filtering")

        # Sort source files before processing
        source_files = sort_by(source_files, base_path=str(source_dir), method=sort_method, random_seed=random_seed)

        # Aplly range
        if skip_first_images or select_every_nth != 1:
            if skip_first_images >= n_files:
                raise ValueError(f"Trying to skip {skip_first_images} images, but only {n_files} found")
            source_files = [source_files[i] for i in range(skip_first_images, n_files, select_every_nth)]
        n_files = len(source_files)
        logger.debug(f"{n_files} in the processing range")
        if not image_load_cap:
            image_load_cap = n_files

        # Create a lowercase mapping of reference files for case-insensitive matching
        ref_map = {}
        if ref_dir:
            for f in os.listdir(ref_dir):
                if (ref_dir / f).is_file():
                    ref_map[Path(f).stem.lower()] = f

        remain = 0
        for index, filename in enumerate(source_files):
            p_filename = Path(filename)
            stem = p_filename.stem
            ext = p_filename.suffix.lower()

            # Determine the destination filename and path
            dest_extension = f".{dest_ext}" if dest_ext else ext
            dest_filename = f"{stem}{dest_extension}"
            dest_path = dest_dir / dest_filename

            # Skip if the result file already exists
            if dest_path.exists():
                continue

            # Find the reference file (case-insensitive and extension-agnostic)
            ref_filename = ""
            if ref_dir:
                ref_filename_found = ref_map.get(stem.lower())
                if ref_filename_found:
                    ref_filename = str(ref_dir / ref_filename_found)

            # Add the absolute paths to the lists
            images.append(str(source_dir / filename))
            results.append(str(dest_path))
            references.append(ref_filename if ref_dir else "")

            if len(images) >= image_load_cap:
                remain = len(source_files) - index - 1
                break

        if not len(images):
            logger.info("All images processed")
            if block_when_finished:
                return ([ExecutionBlocker("No more images to process")], [ExecutionBlocker(None)], [ExecutionBlocker(None)])
            else:
                return ([None], [None], [None])

        cur_len = len(images)
        total = n_files
        last = total-remain
        first = last-cur_len+1

        if cur_len > 1:
            logger.info(f"Listing {cur_len} images out of {n_files} [{first} to {last}] "
                        f"[{(first-1)/total:.0%}-{(last)/total:.0%}] left: {remain}")
        else:
            logger.info(f"Image {first}/{n_files} [{(first-1)/total:.0%}-{(last)/total:.0%}] left: {remain}")
        logger.debug(images)

        return (images, results, references)


class MaskDifference(ComfyNodeABC):
    """
    Compares two MASKs (grayscale images).
    The output is a color IMAGE visualizing the difference.

    Modes:
    1. Simple (Red/Green): Shows added intensity in green and removed in red.
    2. Coincidence (White): Shows additions in green, removals in red, and
       shared intensity in white/grayscale.
    """

    MODES = ["Simple (Red/Green)", "Coincidence (White)"]

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "result": (IO.MASK,),
                "reference": (IO.MASK,),
                "mode": (s.MODES,),
            },
        }

    RETURN_TYPES = (IO.IMAGE,)
    FUNCTION = "generate_diff"
    CATEGORY = BASE_CATEGORY + "/" + "Compare"
    UNIQUE_NAME = "SET_MaskDifference"
    DISPLAY_NAME = "Mask Difference"

    def generate_diff(self, result: torch.Tensor, reference: torch.Tensor, mode: str):
        # Ensure batch sizes match by taking the smaller of the two
        batch_size = min(result.shape[0], reference.shape[0])
        m1 = reference[:batch_size]
        m2 = result[:batch_size]

        # Calculate the core difference
        difference = m2 - m1

        # --- Conditional logic based on the selected mode ---
        if mode == "Simple (Red/Green)":
            # Red channel for removed intensity
            red_channel = torch.clamp(-difference, min=0)
            # Green channel for added intensity
            green_channel = torch.clamp(difference, min=0)
            # Blue channel is all zeros
            blue_channel = torch.zeros_like(red_channel)

        elif mode == "Coincidence (White)":
            # Find the shared intensity
            coincidence = torch.min(m1, m2)
            # Red channel = removed intensity + shared intensity
            red_channel = torch.clamp(-difference, min=0) + coincidence
            # Green channel = added intensity + shared intensity
            green_channel = torch.clamp(difference, min=0) + coincidence
            # Blue channel = shared intensity
            blue_channel = coincidence

        # Stack the R, G, B channels along the last dimension to create
        # the (B, H, W, C) format required for a ComfyUI IMAGE.
        diff_image_bhwc = torch.stack([red_channel, green_channel, blue_channel], dim=-1)

        # Clamp final image tensor to the valid [0.0, 1.0] range
        diff_image_bhwc = torch.clamp(diff_image_bhwc, 0.0, 1.0)

        return (diff_image_bhwc,)


class SaliencyEvaluationMetrics(ComfyNodeABC):
    """
    Computes popular metrics to evaluate Salient Object Detection methods.
    """
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "prediction": (IO.MASK,),
                "ground_truth": (IO.MASK,),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
            },
            "optional": {
                "img_name": (IO.STRING, {"forceInput": True, "tooltip": "Name used as base to save the parameters"}),
                "normalize": (IO.BOOLEAN, {"default": False, "tooltip": "Normalize input masks to be in the [0, 1] range"}),
                "result_save": (IO.BOOLEAN, {"default": False, "tooltip": "Save computed values to IMG_NAME.csv"}),
                "mae_enable": (IO.BOOLEAN, {"default": True, "tooltip": "Compute the MAE"}),
                "mae_save": (IO.BOOLEAN, {"default": False, "tooltip": "Save the MAE using IMG_NAME_MAE.csv"}),
                "max_f_mes_enable": (IO.BOOLEAN, {"default": True, "tooltip": "Compute the Max_F-measure"}),
                "max_f_mes_save": (IO.BOOLEAN, {"default": False, "tooltip": "Save the F-measure using IMG_NAME_F.csv"}),
                "s_mes_enable": (IO.BOOLEAN, {"default": True, "tooltip": "Compute the S-measure"}),
                "s_mes_save": (IO.BOOLEAN, {"default": False, "tooltip": "Save the S-measure using IMG_NAME_S.csv"}),
                "e_mes_enable": (IO.BOOLEAN, {"default": True, "tooltip": "Compute the E-measure"}),
                "e_mes_save": (IO.BOOLEAN, {"default": False, "tooltip": "Save the E-measure using IMG_NAME_E.csv"}),
                "wf_mes_enable": (IO.BOOLEAN, {"default": True, "tooltip": "Compute the Weighted F-measure"}),
                "wf_mes_save": (IO.BOOLEAN, {"default": False, "tooltip":
                                "Save the Weighted F-measure using IMG_NAME_wF.csv"}),
            },
        }

    RETURN_TYPES = ("DICT", IO.STRING, IO.FLOAT, IO.FLOAT, IO.FLOAT, IO.FLOAT, IO.FLOAT)
    RETURN_NAMES = ("all", "img_name", "MAE", "Max_F-measure", "S-measure", "E-measure", "Weighted_F-measure")
    OUTPUT_IS_LIST = (True, True, False, False, False, False, False)
    INPUT_IS_LIST = True
    # We generate files when save is enabled and we want to get information even when nothing is connected to the outputs
    OUTPUT_NODE = True
    FUNCTION = "evaluate"
    CATEGORY = BASE_CATEGORY + "/" + "Analysis"
    UNIQUE_NAME = "SET_SaliencyEvaluationMetrics"
    DISPLAY_NAME = "Saliency Evaluation Metrics"

    def evaluate(self, prediction: torch.Tensor, ground_truth: torch.Tensor, unique_id,
                 img_name, normalize, result_save: bool = False, mae_enable: bool = True, mae_save: bool = False,
                 max_f_mes_enable: bool = True, max_f_mes_save: bool = False, s_mes_enable: bool = True,
                 s_mes_save: bool = False, e_mes_enable: bool = True, e_mes_save: bool = False,
                 wf_mes_enable: bool = True, wf_mes_save: bool = True):
        # Flatten arguments that aren't really expected to be lists
        normalize = normalize[0]
        mae_enable = mae_enable[0]
        mae_save = mae_save[0]
        max_f_mes_enable = max_f_mes_enable[0]
        max_f_mes_save = max_f_mes_save[0]
        s_mes_enable = s_mes_enable[0]
        s_mes_save = s_mes_save[0]
        e_mes_enable = e_mes_enable[0]
        e_mes_save = e_mes_save[0]
        wf_mes_enable = wf_mes_enable[0]
        wf_mes_save = wf_mes_save[0]
        unique_id = unique_id[0]

        device = get_default_comfy_device()

        # Ensure we have lists of the same length
        gt_len = len(ground_truth)
        pred_len = len(prediction)
        if gt_len != pred_len:
            raise ValueError(f"Got {pred_len} predictions and {gt_len} ground thruths, they must match")

        # Find how many images we have
        imgs_len = sum((i.shape[0] for i in prediction))

        # Initialize accumulators for metrics
        mae_total = f_measure_max_total = s_measure_total = e_measure_total = weighted_f_total = 0
        e_measure_max_total = e_measure_adp_total = 0
        all = []

        # Names counter
        index_name = 0
        names_len = len(img_name)
        if names_len != imgs_len:
            raise ValueError(f"Got {imgs_len} images and {names_len} names, they must match")

        for index_img in range(pred_len):
            # Ensure tensors are on the same device
            inputs_are_copies = get_canonical_device(prediction[index_img].device) != device
            gt = ground_truth[index_img].to(device)
            pred = prediction[index_img].to(device)

            # Ensure masks are normalized to [0, 1] range
            if normalize:
                gt, skipped = batched_min_max_norm(gt, in_place=inputs_are_copies, ret_status=True)
                if skipped:
                    logger.debug("GT mask already [0, 1]")
                pred, skipped = batched_min_max_norm(pred, in_place=inputs_are_copies, ret_status=True)
                if skipped:
                    logger.debug("Prediction mask already [0, 1]")

            for i in range(gt.shape[0]):
                if img_name[index_name] is None:
                    return ([None], img_name, None, None, None, None, None)
                # Get the next name
                imgp = Path(img_name[index_name])
                index_name += 1
                logger.debug(f"{index_name}) {imgp.name}")

                pred_i = pred[i]
                gt_i = gt[i]
                res = {}

                # 1. Mean Absolute Error (MAE)
                if mae_enable:
                    mae = torch.mean(torch.abs(pred_i - gt_i)).item()
                    logger.debug(f"MAE: {mae}")
                    mae_total += mae
                    res['mae'] = mae
                    if mae_save:
                        with open(Path(imgp.parent, imgp.stem+"_mae.csv"), "wt") as f:
                            f.write(f"MAE\n{mae}")

                # --- Metrics requiring binary ground truth ---
                if max_f_mes_enable or s_mes_enable or e_mes_enable or wf_mes_enable:
                    gt_binary = (gt_i >= 0.5).float()

                # 2. Max F-measure
                if max_f_mes_enable:
                    f_max, all_f, f_adp = get_f_measure(pred_i, gt_binary)
                    f_measure_max_total += f_max
                    logger.debug(f"Fβmax: {f_max}")
                    res['max_f_mes'] = f_max
                    res['adp_f_mes'] = f_adp
                    res['f'] = all_f[1]
                    res['fp'] = all_f[2]
                    res['fr'] = all_f[3]
                    if max_f_mes_save:
                        with open(Path(imgp.parent, imgp.stem+"_F.csv"), "wt") as f:
                            f.write("Threshold, F-measure\n")
                            for fn in all_f:
                                f.write(f"{fn[0]}, {fn[1]}\n")
                            f.write(f"\nMax, {f_max}\n")

                # 3. S-measure
                if s_mes_enable:
                    s_measure = get_s_measure(pred_i, gt_binary)
                    s_measure_total += s_measure
                    logger.debug(f"Sα: {s_measure}")
                    res['s_mes'] = s_measure
                    if s_mes_save:
                        with open(Path(imgp.parent, imgp.stem+"_S.csv"), "wt") as f:
                            f.write(f"S-measure\n{s_measure}")

                # 4. E-measure
                if e_mes_enable:
                    e_mean, e_max, e_adp, all_e, thres = get_e_measure(pred_i, gt_binary)
                    e_measure_total += e_mean
                    e_measure_max_total += e_max
                    e_measure_adp_total += e_adp
                    logger.debug(f"Eϕ: {e_mean} {e_max} {e_adp}")
                    res['e_mes'] = e_mean
                    res['max_e_mes'] = e_max
                    res['adp_e_mes'] = e_adp
                    res['e'] = all_e
                    if e_mes_save:
                        with open(Path(imgp.parent, imgp.stem+"_E.csv"), "wt") as f:
                            f.write("Threshold, E-measure\n")
                            for index, en in enumerate(all_e):
                                f.write(f"{thres[index]}, {en}\n")
                            f.write("\n")
                            f.write(f"Mean, {e_mean}\n")
                            f.write(f"Max, {e_max}\n")
                            f.write(f"Adaptive, {e_adp}\n")

                # 5. Weighted F-measure
                if wf_mes_enable:
                    wf = get_weighted_f_measure(pred_i, gt_binary)
                    weighted_f_total += wf
                    logger.debug(f"Fβw: {wf}")
                    res['wf_mes'] = wf
                    if wf_mes_save:
                        with open(Path(imgp.parent, imgp.stem+"_wF.csv"), "wt") as f:
                            f.write(f"Weighted F-measure\n{wf}")

                if result_save and res:
                    with open(Path(imgp.parent, imgp.stem+".csv"), "wt") as f:
                        keys = [k for k in SOD_NAMES.keys() if k in res] + [k for k in VECT_NAMES if k in res]
                        f.write(','.join(expand_header_keys(keys))+"\n")
                        f.write(','.join(expand_data_row(res, keys))+"\n")

                all.append(res)

        # Average metrics over the batch/es
        mae_avg = mae_total / gt_len
        f_measure_avg = f_measure_max_total / gt_len
        s_measure_avg = s_measure_total / gt_len
        e_measure_avg = e_measure_total / gt_len
        e_measure_max_avg = e_measure_max_total / gt_len
        e_measure_adp_avg = e_measure_adp_total / gt_len
        weighted_f_avg = weighted_f_total / gt_len

        # Show results in the node
        msg = "<table>"
        if mae_enable:
            msg += f"<tr><td>MAE</td><td>{mae_avg:.4f}</td></tr>"
        if max_f_mes_enable:
            msg += f"<tr><td>Fβmax</td><td>{f_measure_avg:.4f}</td></tr>"
        if s_mes_enable:
            msg += f"<tr><td>Sα</td><td>{s_measure_avg:.4f}</td></tr>"
        if e_mes_enable:
            msg += f"<tr><td>Eϕmean</td><td>{e_measure_avg:.4f}</td></tr>"
            msg += f"<tr><td>Eϕmax</td><td>{e_measure_max_avg:.4f}</td></tr>"
            msg += f"<tr><td>Eϕadp</td><td>{e_measure_adp_avg:.4f}</td></tr>"
        if wf_mes_enable:
            msg += f"<tr><td>Fβw</td><td>{weighted_f_avg:.4f}</td></tr>"
        msg += "</table>"
        send_progress_text(unique_id, msg)

        return (all, img_name, mae_avg, f_measure_avg, s_measure_avg, e_measure_avg, weighted_f_avg)


class ConsolidateMetrics(ComfyNodeABC):
    """
    Consolidates new metrics with the ones already computed

    This node reads the `destination` CSV and adds the metrics from `metrics`.
    Each row starts with the `img_name` corresponding for the metrics.
    """
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "metrics": ("DICT",),
                "img_name": (IO.STRING, {"forceInput": True, "tooltip": "File names for the evaluated images"}),
                "destination": (IO.STRING, {
                    "default": "./result",
                    "tooltip": "Path for the result images.\nRelative to ComfyUI output\n"
                               "If this is a directory the file\nwill be named `consolidated.csv` inside it"
                }),
            },
        }

    INPUT_IS_LIST = True
    FUNCTION = "execute"
    CATEGORY = BASE_CATEGORY + "/" + "Analysis"
    UNIQUE_NAME = "SET_ConsolidateMetrics"
    DISPLAY_NAME = "Consolidate Metrics"
    RETURN_TYPES = ("DICT", )
    RETURN_NAMES = ("consolidated", )
    OUTPUT_IS_LIST = (True, )
    OUTPUT_NODE = True

    def compact_row(self, row_data, metric_keys_ordered):
        metric_values = {}
        # Use a manual index to track our position in the full data row (which has many columns).
        data_col_idx = 0

        # Iterate through our "compressed" list of metric keys.
        for key in metric_keys_ordered:
            # Check if the current key corresponds to a vector.
            if key in VECT_NAMES:
                # --- This is a vector metric (like 'f', 'fp', etc.) ---
                # Define the slice of the row that contains the vector data.
                start = data_col_idx
                end = data_col_idx + F_POINTS
                # Extract the string values for this vector.
                vector_str_values = row_data[start:end]
                # Convert the string values to a list of floats.
                vector_float_values = [float(val) for val in vector_str_values]
                # Create a PyTorch tensor and store it in the dictionary.
                metric_values[key] = torch.tensor(vector_float_values, dtype=torch.float32)
                # Advance the column index by the number of points we just consumed.
                data_col_idx += F_POINTS
            else:
                # --- This is a simple scalar metric (like 'mae', 's_mes', etc.) ---
                # Read the single value from the current position.
                scalar_str_value = row_data[data_col_idx]
                # Convert to float and store it in the dictionary.
                metric_values[key] = float(scalar_str_value)
                # Advance the column index by one.
                data_col_idx += 1
        return metric_values

    def execute(self, metrics, img_name, destination):
        # --- 1. Input Validation and Flattening ---

        if metrics[0] is None or img_name[0] is None or destination[0] is None:
            return ({}, )

        if len(metrics) != len(img_name):
            raise ValueError(f"Got {len(metrics)} metrics and {len(img_name)} file names. They must match.")
        if len(destination) != 1:
            raise ValueError("Only one `destination` is accepted.")

        # Resolve the final destination path for the CSV file.
        dest_path = Path(get_output_directory(), destination[0])
        if dest_path.is_dir():
            dest_path = dest_path / 'consolidated.csv'

        # Ensure the parent directory exists.
        dest_path.parent.mkdir(exist_ok=True)

        # --- 2. Load Existing Data from CSV (if it exists) ---

        existing_data = {}
        header = []
        metric_keys_ordered = []

        if dest_path.is_file():
            try:
                with open(dest_path, 'r', newline='') as f:
                    reader = csv.reader(f)

                    # Read the header to preserve column order.
                    header = next(reader)

                    # Extract the internal metric keys from the display names in the header.
                    # This is crucial for correctly mapping new data to the existing columns.
                    metric_keys_ordered = []
                    for i, h in enumerate(header[1:]):
                        res = IS_VECT_NAME.match(h)
                        if res:
                            vect_name = res.group(1)
                            index = res.group(2)
                            if index == "0":
                                metric_keys_ordered.append(vect_name.lower())
                        else:
                            metric_keys_ordered.append(REVERSE_SOD_NAMES.get(h))

                    # Load existing rows, stopping at any blank line (which precedes totals).
                    for row in reader:
                        if not row:  # Stop if we hit a blank line
                            break

                        # The first column is the image name (with quotes).
                        filename = row[0].strip('"')

                        # Create a dictionary for the row's metrics.
                        metric_values = self.compact_row(row[1:], metric_keys_ordered)
                        existing_data[filename] = metric_values

            except (IOError, StopIteration, IndexError, ValueError) as e:
                logger.warning(f"Could not properly read existing file at {dest_path}. It will be overwritten. Error: {e}")
                existing_data = {}  # Reset on read error

        # --- 3. Consolidate New Metrics ---

        # Add or update the new metrics into our dictionary of existing data.
        for i, new_metric_dict in enumerate(metrics):
            filename = Path(img_name[i]).name
            existing_data[filename] = new_metric_dict

        if not existing_data:
            logger.warning("[Warning] No metrics to consolidate. Aborting file write.")
            return ({}, )

        # --- 4. Prepare for Writing (Sort and Define Header if New) ---

        # If the file was new, define the header and key order now.
        if not header:
            # Get the keys from the first available metric dictionary.
            # first_item_keys = list(next(iter(existing_data.values())).keys())
            # metric_keys_ordered = first_item_keys # sorted(first_item_keys)  # Sort for consistent order
            # Create the header with display names.
            metric_keys_ordered = list(SOD_NAMES.keys()) + list(VECT_NAMES)
            header = ["Image"] + expand_header_keys(metric_keys_ordered)

        # Sort the consolidated data alphabetically by filename.
        sorted_filenames = sorted(existing_data.keys())

        # --- 5. Compute New Totals (Averages) ---

        # Use defaultdict to handle missing metrics gracefully.
        totals = defaultdict(float)
        valid_counts = defaultdict(int)

        for filename in sorted_filenames:
            for key, value in existing_data[filename].items():
                totals[key] += value
                valid_counts[key] += 1

        averages = {key: totals[key] / valid_counts[key] for key in metric_keys_ordered if valid_counts[key] > 0}

        # --- 6. Write Consolidated File ---

        with open(dest_path, 'w', newline='') as f:
            writer = csv.writer(f)

            # Write the header.
            writer.writerow(expand_header(header))

            # Write the sorted data rows.
            for filename in sorted_filenames:
                metric_dict = existing_data[filename]
                # Format the filename as required and get metric values in the correct order.
                row_data = [filename] + expand_data_row(metric_dict, metric_keys_ordered)
                writer.writerow(row_data)

            # Write a blank line to separate data from totals.
            writer.writerow([])

            # Write the totals row.
            total_row = ["Total"] + expand_data_row(averages, metric_keys_ordered, formated=True)
            writer.writerow(total_row)

            # Do we have F(th)?
            Fmax = averages.get('f').max() if 'f' in averages else 0.0
            Emax = averages.get('e').max() if 'e' in averages else 0.0

            if Fmax or Fmax:
                writer.writerow([])

            if Fmax:
                # This is the maximum for the average F-measure
                # Is more representative for the dataset than the average of the maximums of each image
                writer.writerow(['Fmax dataset', f"{Fmax:.4f}"])
                Ftot = averages.get('f').sum()
                writer.writerow(['Fmean dataset', f"{Ftot/F_POINTS:.4f}"])

            # Do we have E(th)?
            if Emax:
                writer.writerow(['Emax dataset', f"{Emax:.4f}"])
                Etot = averages.get('e').sum()
                writer.writerow(['Emean dataset', f"{Etot/F_POINTS:.4f}"])

        logger.info(f"Metrics consolidated and saved to {dest_path}")

        return ([v for v in existing_data.values()], )


class PlotMetricCurvesPIL(ComfyNodeABC):
    """
    Plots the Precision vs Recall and F-measure curves

    Generates two images containing the curves from the consolidated `metrics`
    """
    # Define available colors for the plot line
    COLORS = ['blue', 'green', 'red', 'cyan', 'magenta', 'black']

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "metrics": ("DICT",),
                "plot_title": (IO.STRING, {"default": "Saliency Evaluation"}),
                "curve_color": (s.COLORS,),
                "width": (IO.INT, {"default": 800, "min": 256, "max": 4096}),
                "height": (IO.INT, {"default": 600, "min": 256, "max": 4096}),
            },
        }

    INPUT_IS_LIST = True
    RETURN_TYPES = (IO.IMAGE, IO.IMAGE)
    RETURN_NAMES = ("pr_curve_plot", "fm_curve_plot")
    FUNCTION = "execute"
    CATEGORY = BASE_CATEGORY + "/" + "Analysis"
    UNIQUE_NAME = "SET_PlotMetricCurvesPIL"
    DISPLAY_NAME = "Plot SOD metric curves"

    def _pil_to_tensor(self, pil_image):
        """Converts a PIL Image to a ComfyUI-compatible IMAGE tensor."""
        # Convert to numpy array, normalize to [0, 1], and add batch dimension
        return torch.from_numpy(np.array(pil_image).astype(np.float32) / 255.0).unsqueeze(0)

    def no1_create_plot_with_pil(self, data_x, data_y, title, x_label, y_label, width, height, color, x_lim, y_lim):
        """
        A helper function to generate a plot from scratch using PIL.
        """
        # --- 1. Setup Canvas and Drawing Tools ---
        padding = 60  # Pixels for labels and ticks
        img = Image.new('RGB', (width, height), 'white')
        draw = ImageDraw.Draw(img)

        # Try to load a standard font, fall back to default if not found
        try:
            font = ImageFont.truetype("arial.ttf", 25)
            title_font = ImageFont.truetype("arial.ttf", 30)
        except IOError:
            font = ImageFont.load_default()
            title_font = font

        # --- 2. Define Plot Area and Coordinate Mapping ---
        plot_width = width - padding * 2
        plot_height = height - padding * 2

        # Function to map data coordinates to pixel coordinates
        def to_pixel(x, y):
            px = padding + ((x - x_lim[0]) / (x_lim[1] - x_lim[0])) * plot_width
            # Y is inverted in PIL (0 is at the top)
            py = (height - padding) - ((y - y_lim[0]) / (y_lim[1] - y_lim[0])) * plot_height
            return int(px), int(py)

        # --- 3. Draw Grid, Axes, and Ticks ---
        num_grid_lines = 5
        # Vertical grid lines and X-axis ticks
        for i in range(num_grid_lines + 1):
            val = x_lim[0] + (i / num_grid_lines) * (x_lim[1] - x_lim[0])
            px, _ = to_pixel(val, y_lim[0])
            draw.line([(px, padding), (px, height - padding)], fill=(220, 220, 220), width=1)
            label = f"{val:.1f}" if val % 1 else str(int(val))
            draw.text((px - 10, height - padding + 5), label, font=font, fill='black')

        # Horizontal grid lines and Y-axis ticks
        for i in range(num_grid_lines + 1):
            val = y_lim[0] + (i / num_grid_lines) * (y_lim[1] - y_lim[0])
            _, py = to_pixel(x_lim[0], val)
            draw.line([(padding, py), (width - padding, py)], fill=(220, 220, 220), width=1)
            label = f"{val:.1f}"
            draw.text((padding - 35, py - 8), label, font=font, fill='black')

        # Draw main axes lines
        draw.line([(padding, height - padding), (width - padding, height - padding)], fill='black', width=2)  # X-axis
        draw.line([(padding, padding), (padding, height - padding)], fill='black', width=2)  # Y-axis

        # --- 4. Draw the Data Curve ---
        pixel_points = [to_pixel(x, y) for x, y in zip(data_x, data_y)
                        if x_lim[0] <= x <= x_lim[1] and y_lim[0] <= y <= y_lim[1]]
        if len(pixel_points) > 1:
            draw.line(pixel_points, fill=color, width=3)

        # --- 5. Draw Title and Labels ---
        title_bbox = draw.textbbox((0, 0), title, font=title_font)
        draw.text(((width - (title_bbox[2] - title_bbox[0])) / 2, 10), title, font=title_font, fill='black')

        xlabel_bbox = draw.textbbox((0, 0), x_label, font=font)
        draw.text(((width - (xlabel_bbox[2] - xlabel_bbox[0])) / 2, height - 25), x_label, font=font, fill='black')

        ylabel_bbox = draw.textbbox((0, 0), y_label, font=font)
        # For Y label, we draw it rotated by drawing character by character
        y_label_y_start = (height + (ylabel_bbox[3] - ylabel_bbox[1])) / 2
        for i, char in enumerate(y_label):
            draw.text((10, y_label_y_start - i*15), char, font=font, fill='black')

        return img

    def _get_nice_limits(self, data_min, data_max, padding_percent=0.05):
        """Calculates 'nice' axis limits with padding, handling edge cases."""
        # Handle the case where all data points are the same
        if data_min == data_max:
            return data_min - 0.1, data_max + 0.1

        # Calculate padding based on the data range
        data_range = data_max - data_min
        padding = data_range * padding_percent

        # Return the padded limits
        return data_min - padding, data_max + padding

    def _create_plot_with_pil(self, data_x, data_y, title, x_label, y_label, width, height, color, x_lim=None, y_lim=None):
        """
        Generates a plot from scratch using PIL with auto-scaling and rotated Y-axis text.
        """
        # --- 1. Auto-scale limits if they are not provided ---
        if x_lim is None:
            x_lim = self._get_nice_limits(np.min(data_x), np.max(data_x))
        if y_lim is None:
            y_lim = self._get_nice_limits(np.min(data_y), np.max(data_y))

        # --- 2. Setup Canvas and Drawing Tools ---
        padding_left = 80  # Increased padding to accommodate rotated label
        padding_right = 30
        padding_top = 60
        padding_bottom = 60

        img = Image.new('RGB', (width, height), 'white')
        draw = ImageDraw.Draw(img)
        font = load_font("Arial", 15)
        title_font = load_font("Arial", 20)

        # --- 3. Define Plot Area and Coordinate Mapping ---
        # (This section is unchanged)
        plot_width = width - padding_left - padding_right
        plot_height = height - padding_top - padding_bottom

        def to_pixel(x, y):
            px = padding_left + ((x - x_lim[0]) / (x_lim[1] - x_lim[0])) * plot_width
            py = (height - padding_bottom) - ((y - y_lim[0]) / (y_lim[1] - y_lim[0])) * plot_height
            return int(px), int(py)

        # --- 4. Draw Grid, Axes, and Ticks ---
        # (This section is unchanged)
        num_grid_lines = 5
        for i in range(num_grid_lines + 1):
            val_x = x_lim[0] + (i / num_grid_lines) * (x_lim[1] - x_lim[0])
            px, _ = to_pixel(val_x, y_lim[0])
            draw.line([(px, padding_top), (px, height - padding_bottom)], fill=(220, 220, 220), width=1)
            label = f"{val_x:.2f}" if val_x % 1 else str(int(val_x))
            draw.text((px, height - padding_bottom + 5), label, font=font, fill='black', anchor="mt")

        for i in range(num_grid_lines + 1):
            val_y = y_lim[0] + (i / num_grid_lines) * (y_lim[1] - y_lim[0])
            _, py = to_pixel(x_lim[0], val_y)
            draw.line([(padding_left, py), (width - padding_right, py)], fill=(220, 220, 220), width=1)
            label = f"{val_y:.2f}"
            draw.text((padding_left - 10, py), label, font=font, fill='black', anchor="rm")

        draw.line([(padding_left, height - padding_bottom), (width - padding_right, height - padding_bottom)], fill='black',
                  width=2)
        draw.line([(padding_left, padding_top), (padding_left, height - padding_bottom)], fill='black', width=2)

        # --- 5. Draw the Data Curve ---
        # (This section is unchanged)
        pixel_points = [to_pixel(x, y) for x, y in zip(data_x, data_y) if x_lim[0] <= x <= x_lim[1] and
                        y_lim[0] <= y <= y_lim[1]]
        if len(pixel_points) > 1:
            draw.line(pixel_points, fill=color, width=3)

        # --- 6. Draw Title and Labels ---

        # Draw Title and X-axis Label (unchanged)
        draw.text((width / 2, padding_top / 2), title, font=title_font, fill='black', anchor="mm")
        draw.text((width / 2, height - padding_bottom / 4), x_label, font=title_font, fill='black', anchor="mb")

        # --- Draw Rotated Y-axis Label ---

        # a. Get the size of the unrotated text
        # y_label_bbox = font.getbbox(y_label)
        # y_label_width = y_label_bbox[2] - y_label_bbox[0]
        # y_label_height = y_label_bbox[3] - y_label_bbox[1]

        # b. Create a new, transparent canvas for the text
        # txt_canvas = Image.new('RGBA', (y_label_width, y_label_height), (0, 0, 0, 0))
        txt_canvas = Image.new('RGBA', (height, padding_left), (0, 0, 0, 0))
        txt_draw = ImageDraw.Draw(txt_canvas)

        # c. Draw the text onto the temporary canvas
        txt_draw.text((height // 2, padding_left // 4), y_label, font=title_font, fill='black', anchor="mb")

        # d. Rotate the text canvas by 90 degrees
        # 'expand=True' makes the new image large enough to hold the rotated content
        rotated_y_label = txt_canvas.rotate(90, expand=True)

        # e. Calculate the paste position on the main canvas
        # Center it vertically in the plot area and horizontally in the left padding area
        paste_x = 0  # int((padding_left - rotated_y_label.width) / 2)
        paste_y = int((height - rotated_y_label.height) / 2)

        # f. Paste the rotated text onto the main image, using its alpha channel as a mask
        img.paste(rotated_y_label, (paste_x, paste_y), rotated_y_label)

        return img

    def execute(self, metrics, plot_title, curve_color, width, height):
        # --- 1. Aggregate Data (same as matplotlib version) ---
        plot_title_str = plot_title[0]
        curve_color_str = curve_color[0]

        if not metrics:
            logger.warning("No metrics data provided. Returning blank images.")
            blank_image = self._pil_to_tensor(Image.new('RGB', (width[0], height[0]), 'white'))
            return (blank_image, blank_image)

        all_precisions, all_recalls, all_fmeasures = [], [], []

        for metric_dict in metrics:
            # Check if the dictionary contains the compressed tensor keys.
            if 'fp' in metric_dict and 'fr' in metric_dict and 'f' in metric_dict:
                # Retrieve the tensor, move to CPU, and convert to a NumPy array.
                # The .cpu() is important for safety in case tensors are on the GPU.
                all_precisions.append(metric_dict['fp'].cpu().numpy())
                all_recalls.append(metric_dict['fr'].cpu().numpy())
                all_fmeasures.append(metric_dict['f'].cpu().numpy())

        if not all_precisions:
            logger.warning("Metrics data did not contain F/FP/FR keys. Returning blank images.")
            blank_image = self._pil_to_tensor(Image.new('RGB', (width[0], height[0]), 'white'))
            return (blank_image, blank_image)

        avg_precision = np.mean(all_precisions, axis=0)
        avg_recall = np.mean(all_recalls, axis=0)
        avg_fmeasure = np.mean(all_fmeasures, axis=0)

        # --- 2. Generate Precision-Recall (PR) Curve Plot ---
        pr_plot_pil = self._create_plot_with_pil(
            data_x=avg_recall, data_y=avg_precision,
            title=f"{plot_title_str} (PR Curve)", x_label="Recall", y_label="Precision",
            width=width[0], height=height[0], color=curve_color_str,
            x_lim=(np.min(avg_recall), np.max(avg_recall)), y_lim=None  # (np.min(avg_precision), np.max(avg_precision)
        )
        pr_plot_tensor = self._pil_to_tensor(pr_plot_pil)

        # --- 3. Generate F-Measure Curve Plot ---
        threshold_axis = np.arange(F_POINTS)
        fm_plot_pil = self._create_plot_with_pil(
            data_x=threshold_axis, data_y=avg_fmeasure,
            title=f"{plot_title_str} (F-Measure Curve)", x_label="Threshold", y_label="F-measure",
            width=width[0], height=height[0], color=curve_color_str,
            x_lim=(0, F_POINTS), y_lim=(0.0, 1.0)
        )
        fm_plot_tensor = self._pil_to_tensor(fm_plot_pil)

        return (pr_plot_tensor, fm_plot_tensor)


class CompositeFace(ComfyNodeABC):
    """
    Inserts the `animated` face in the `reference` images at the `bboxes` coordinates.

    Composite (paste) animated face crops back onto reference images.
    It handles a M-to-N relationship, where M reference images and bboxes correspond
    to M*N animated face images.
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "animated": (IO.IMAGE,),      # The M*N batch of cropped faces
                "reference": (IO.IMAGE,),     # The M batch of original context images
                "bboxes": ("BBOX",),         # The M list of (x, y, w, h) tuples
            },
        }

    RETURN_TYPES = (IO.IMAGE,)
    RETURN_NAMES = ("images",)
    FUNCTION = "composite"

    CATEGORY = BASE_CATEGORY + "/" + MANIPULATION_CATEGORY
    UNIQUE_NAME = "SET_CompositeFace"
    DISPLAY_NAME = "Face Composite"

    def composite(self, animated: torch.Tensor, reference: torch.Tensor, bboxes: list):
        # 1. Get batch sizes and validate the M vs M*N relationship
        ref_count = reference.shape[0]
        anim_count = animated.shape[0]
        bbox_count = len(bboxes)

        if ref_count == 0 or anim_count == 0:
            logger.info("Warning: One of the input image batches is empty. Returning empty tensor.")
            return (torch.zeros((0, 1, 1, 3)),)

        if ref_count != bbox_count:
            raise ValueError(f"Mismatch: Received {ref_count} reference images but {bbox_count} bboxes. "
                             "These must be equal.")

        if anim_count % ref_count != 0:
            raise ValueError(f"Batch size mismatch: The 'animated' batch ({anim_count}) is not a multiple of the "
                             f"'reference' batch ({ref_count}).")

        # N: Number of animated frames per reference image
        n_frames_per_ref = anim_count // ref_count
        logger.info(f"Processing {ref_count} reference images, each with {n_frames_per_ref} animated frames.")

        output_images = []

        # 2. Iterate through each reference image and its corresponding bbox
        for i in range(ref_count):
            ref_tensor = reference[i]
            bbox = bboxes[i]

            # The bbox from your code is the area to be replaced.
            # Assuming it's in the format (x, y, width, height)
            try:
                x, y, w, h = map(int, bbox)
            except (ValueError, TypeError) as e:
                raise TypeError(f"Bbox item {i} has an invalid format: {bbox}. Expected (x, y, w, h). Error: {e}")

            # Convert the reference image to Pillow Image ONCE before the inner loop
            ref_pil = tensor_to_pil(ref_tensor)

            # 3. For this one reference, loop through its N animated frames
            for j in range(n_frames_per_ref):
                anim_index = i * n_frames_per_ref + j
                anim_tensor = animated[anim_index]

                # Convert the small animated face to Pillow Image
                anim_pil = tensor_to_pil(anim_tensor)

                # 4. Resize the animated face to fit the target bbox
                # Image.Resampling.LANCZOS is a high-quality resampling filter.
                resized_anim_face = anim_pil.resize((w, h), Image.Resampling.LANCZOS)

                # 5. Perform the paste operation.
                # It's CRITICAL to work on a copy of the reference image for each frame.
                pasted_image_pil = ref_pil.copy()
                pasted_image_pil.paste(resized_anim_face, (x, y))

                # 6. Convert the final image back to a tensor and add to the output list
                final_tensor = pil_to_tensor(pasted_image_pil)
                output_images.append(final_tensor)

        # 7. Stack all the generated images into a single batch tensor
        if not output_images:
            return (torch.zeros_like(reference),)  # Return something if all pastes failed

        final_batch = torch.stack(output_images)

        return (final_batch,)


class CompositeFaceFrameByFrame(CompositeFace):
    """
    Inserts the `animated` face in the `reference` video at the `bboxes` coordinates.

    Composite animated frames onto reference frames on a 1-to-1 basis.
    It expects the 'animated' and 'reference' batches to have the same number of frames.
    It uses the *first* bounding box from the 'bboxes' input for all frames.
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "animated": (IO.IMAGE,),      # The batch of cropped/processed frames
                "reference": (IO.IMAGE,),     # The batch of original frames
                "bboxes": ("BBOX",),         # A list of bboxes; only the first is used
            },
        }

    CATEGORY = BASE_CATEGORY + "/" + MANIPULATION_CATEGORY
    UNIQUE_NAME = "SET_CompositeFaceFrameByFrame"
    DISPLAY_NAME = "Face Composite (frame by frame)"

    def composite(self, animated: torch.Tensor, reference: torch.Tensor, bboxes: list):
        # 1. Validate inputs
        anim_count = animated.shape[0]
        ref_count = reference.shape[0]

        if anim_count != ref_count:
            raise ValueError(f"Batch size mismatch: Received {anim_count} animated frames and {ref_count} reference "
                             "frames. They must be equal.")

        if not bboxes:
            raise ValueError("Bboxes input is empty. A bounding box is required.")

        # 2. Extract the single bounding box to be used for all frames
        if len(bboxes) > 1:
            logger.info(f"Warning: Received {len(bboxes)} bboxes. Using only the first one for all frames.")

        try:
            # Use the first bbox from the list
            x, y, w, h = map(int, bboxes[0])
            bbox_to_use = (x, y, w, h)
        except (ValueError, TypeError) as e:
            raise TypeError(f"The first bbox has an invalid format: {bboxes[0]}. Expected (x, y, w, h). Error: {e}")

        logger.info(f"Compositing {anim_count} frames using static bbox: {bbox_to_use}")

        output_images = []

        # 3. Loop through each frame in a 1-to-1 fashion
        for i in range(anim_count):
            ref_tensor = reference[i]
            anim_tensor = animated[i]

            # Convert tensors to Pillow Images
            ref_pil = tensor_to_pil(ref_tensor)
            anim_pil = tensor_to_pil(anim_tensor)

            # 4. Resize the animated face to fit the bbox
            # Image.Resampling.LANCZOS is a high-quality filter comparable to OpenCV's INTER_CUBIC/LANCZOS4
            resized_anim_face = anim_pil.resize((w, h), Image.Resampling.LANCZOS)

            # 5. Perform the paste operation
            # Pillow's paste is simpler. It handles coordinates and requires a copy.
            pasted_image_pil = ref_pil.copy()
            pasted_image_pil.paste(resized_anim_face, (x, y))

            # 6. Convert back to tensor and add to output list
            final_tensor = pil_to_tensor(pasted_image_pil)
            output_images.append(final_tensor)

        # 7. Stack all images into the final output batch
        final_batch = torch.stack(output_images)

        return (final_batch,)


class NormalizeToImageNetDataset(ComfyNodeABC):
    """
    Normalize the image to the ImageNet dataset

    Note that this is statistical.
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": (IO.IMAGE,),
            },
        }
    RETURN_TYPES = (IO.IMAGE,)
    RETURN_NAMES = ("image",)
    FUNCTION = "normalize"
    CATEGORY = BASE_CATEGORY + "/" + NORMALIZATION
    UNIQUE_NAME = "SET_NormalizeToImageNetDataset"
    DISPLAY_NAME = "Normalize Image to ImageNet"

    def normalize(self, image: torch.Tensor):
        return (TF.normalize(image.permute(0, 3, 1, 2),  # BHWC -> BCHW
                             mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]).permute(0, 2, 3, 1),)  # BCHW -> BHWC


class NormalizeToRangeMinus05to05(ComfyNodeABC):
    """
    Normalize the image to the [-0.5, 0.5] range

    Note that this is statistical.
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"image": (IO.IMAGE,), }, }
    RETURN_TYPES = (IO.IMAGE,)
    RETURN_NAMES = ("image",)
    FUNCTION = "normalize"
    CATEGORY = BASE_CATEGORY + "/" + NORMALIZATION
    UNIQUE_NAME = "SET_NormalizeToRangeMinus05to05"
    DISPLAY_NAME = "Normalize Image to [-0.5, 0.5]"

    def normalize(self, image: torch.Tensor):
        return (TF.normalize(image.permute(0, 3, 1, 2),  # BHWC -> BCHW
                             mean=[0.5, 0.5, 0.5],
                             std=[1.0, 1.0, 1.0]).permute(0, 2, 3, 1),)  # BCHW -> BHWC


class NormalizeToRangeMinus1to1(ComfyNodeABC):
    """
    Normalize the image to the [-1, 1] range

    Note that this is statistical.
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"image": (IO.IMAGE,), }, }
    RETURN_TYPES = (IO.IMAGE,)
    RETURN_NAMES = ("image",)
    FUNCTION = "normalize"
    CATEGORY = BASE_CATEGORY + "/" + NORMALIZATION
    UNIQUE_NAME = "SET_NormalizeToRangeMinus1to1"
    DISPLAY_NAME = "Normalize Image to [-1, 1] (i.e. GAN)"

    def normalize(self, image: torch.Tensor):
        return (TF.normalize(image.permute(0, 3, 1, 2),  # BHWC -> BCHW
                             mean=[0.5, 0.5, 0.5],
                             std=[0.5, 0.5, 0.5]).permute(0, 2, 3, 1),)  # BCHW -> BHWC


class NormalizeArbitrary(ComfyNodeABC):
    """
    Nnormalize the image to arbitrary mean/std, provided in `parameters`

    Note that this is statistical.
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": (IO.IMAGE,),
                "parameters": ("NORM_PARAMS",),
            },
        }
    RETURN_TYPES = (IO.IMAGE,)
    RETURN_NAMES = ("image",)
    FUNCTION = "normalize"
    CATEGORY = BASE_CATEGORY + "/" + NORMALIZATION
    UNIQUE_NAME = "SET_NormalizeArbitrary"
    DISPLAY_NAME = "Arbitrary Normalize"

    def normalize(self, image: torch.Tensor, parameters):
        return (TF.normalize(image.movedim(-1, 1),  # BHWC -> BCHW
                             mean=parameters["mean"],
                             std=parameters["std"]).movedim(1, -1),)  # BCHW -> BHWC


class NormalizeParameters(ComfyNodeABC):
    """ Parameters for the arbitrary normalization """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mean_red": NORM_PARAM,
                "mean_green": NORM_PARAM,
                "mean_blue": NORM_PARAM,
                "std_red": NORM_PARAM,
                "std_green": NORM_PARAM,
                "std_blue": NORM_PARAM,
            },
        }
    RETURN_TYPES = ("NORM_PARAMS",)
    RETURN_NAMES = ("parameters",)
    FUNCTION = "normalize"
    CATEGORY = BASE_CATEGORY + "/" + NORMALIZATION
    UNIQUE_NAME = "SET_NormalizeParameters"
    DISPLAY_NAME = "Normalize Parameters"

    def normalize(self, mean_red, mean_green, mean_blue, std_red, std_green, std_blue):
        return ({"mean": [mean_red, mean_green, mean_blue], "std": [std_red, std_green, std_blue]},)


class ApplyMaskAFFCE(ComfyNodeABC):
    """
    Apply a mask to an image using `Approximate Fast Foreground Colour Estimation`.
    See: https://github.com/Photoroom/fast-foreground-estimation
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": (IO.IMAGE,),
                "masks": (IO.MASK,),
                "blur_size": BLUR_SIZE_OPT,
                "blur_size_two": BLUR_SIZE_TWO_OPT,
                "fill_color": (IO.BOOLEAN, {
                    "default": False,
                    "tooltip": ("Fill the background using a color.\n"
                                "Returns an RGB image, otherwise an RGBA.")
                }),
                "color": COLOR_OPT,
                "batched":  (IO.BOOLEAN, {
                    "default": True,
                    "tooltip": ("Process the images at once.\n"
                                "Faster, needs more memory")
                }),
            }
        }

    RETURN_TYPES = (IO.IMAGE, IO.MASK,)
    RETURN_NAMES = ("image", "mask",)
    FUNCTION = "get_foreground"
    CATEGORY = BASE_CATEGORY + "/" + MANIPULATION_CATEGORY
    UNIQUE_NAME = "SET_ApplyMaskAFFCE"
    DISPLAY_NAME = "Apply Mask using AFFCE"

    def get_foreground(self, images, masks, blur_size=91, blur_size_two=7, fill_color=False, color=None, batched=True):
        out_images = apply_mask(logger, images, masks, get_default_comfy_device(), blur_size, blur_size_two,
                                fill_color, color, batched)
        return out_images.cpu(), masks.cpu()


class AFFCE(ComfyNodeABC):
    """
    Estimate the foreground image using `Approximate Fast Foreground Colour Estimation`.
    See: https://github.com/Photoroom/fast-foreground-estimation
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": (IO.IMAGE,),
                "masks": (IO.MASK,),
                "blur_size": BLUR_SIZE_OPT,
                "blur_size_two": BLUR_SIZE_TWO_OPT,
                "batched":  (IO.BOOLEAN, {
                    "default": True,
                    "tooltip": ("Process the images at once.\n"
                                "Faster, needs more memory")
                }),
            }
        }

    RETURN_TYPES = (IO.IMAGE, IO.MASK,)
    RETURN_NAMES = ("foreground", "mask",)
    FUNCTION = "get_foreground"
    CATEGORY = BASE_CATEGORY + "/" + FOREGROUND
    UNIQUE_NAME = "SET_AFFCE"
    DISPLAY_NAME = "Estimate foreground (AFFCE)"

    def get_foreground(self, images, masks, blur_size=91, blur_size_two=7, batched=True):
        device = get_default_comfy_device()
        images_on_device = images.to(device)
        masks_on_device = masks.to(device)

        out_images = affce(images_on_device, masks_on_device, r1=blur_size, r2=blur_size_two, batched=batched)

        return out_images.cpu(), masks.cpu()


class FMLFE(ComfyNodeABC):
    """
    Estimate the foreground image using `Fast Multi-Level Foreground Estimation`

    Uses the Fast Multi-Level Foreground Estimation algorithm
    to produce a high-quality foreground and background separation.
    It can intelligently select the best available backend (CuPy, OpenCL, Numba, or PyTorch).
    """
    @classmethod
    def INPUT_TYPES(cls):
        # Create the dropdown list for the implementation choice
        impl_list = ['auto'] + IMPL_PRIORITY

        return {
            "required": {
                "images": (IO.IMAGE, {
                    "tooltip": "The source image(s) from which to estimate the foreground and background."
                }),
                "masks": (IO.MASK, {
                    "tooltip": "The alpha matte that guides the estimation. White areas are treated as known "
                               "foreground, black as known background, and gray areas are the semi-transparent "
                               "regions the algorithm will solve for."
                }),
                "implementation": (impl_list, {
                    "default": "auto",
                    "tootip": "Select the computation backend. 'auto' mode will automatically try to use the "
                              "fastest available implementation, in order of priority: CuPy (NVIDIA GPU), "
                              "OpenCL (GPU), Numba (CPU/GPU), and finally the pure PyTorch version."
                }),
            },
            "optional": {
                "regularization": (IO.FLOAT, {
                    "default": 1e-5,
                    "min": 0.0,
                    "max": 0.1,
                    "step": 1e-5,
                    "display": "number",
                    "tooltip": "The regularization strength (epsilon). This acts as a smoothness prior. "
                               "Higher values result in smoother, more blended foreground and background colors, "
                               "but may lose very fine details. Lower values preserve more detail but can be noisier."
                }),
                "n_small_iterations": (IO.INT, {
                    "default": 10,
                    "min": 1,
                    "max": 100,
                    "tooltip": "The number of solver iterations to perform on the lower-resolution levels of the "
                               "image pyramid. More iterations can improve quality at the cost of speed."
                }),
                "n_big_iterations": (IO.INT, {
                    "default": 2,
                    "min": 1,
                    "max": 100,
                    "tooltip": "The number of solver iterations to perform on the higher-resolution (larger) levels "
                               "of the image pyramid. Fewer iterations are typically needed at high resolution as the "
                               "details are propagated up from the smaller levels."
                }),
                "small_size": (IO.INT, {
                    "default": 32,
                    "min": 8,
                    "max": 256,
                    "tooltip": "The pixel dimension threshold. Image pyramid levels smaller than this size will use "
                               "the higher 'n_small_iterations' count, while larger levels will use 'n_big_iterations'."
                }),
                "gradient_weight": (IO.FLOAT, {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 10.0,
                    "step": 0.1,
                    "tooltip": "Controls how strongly the edges in the alpha matte influence color blending. "
                               "A higher value makes the algorithm respect the mask's edges more, leading to sharper "
                               "color boundaries. A lower value allows more color bleeding, an effect similar to "
                               "increasing regularization."
                }),
            }
        }

    RETURN_TYPES = (IO.IMAGE, IO.IMAGE, IO.MASK,)
    RETURN_NAMES = ("foreground", "background", "mask")
    FUNCTION = "estimate"
    CATEGORY = BASE_CATEGORY + "/" + FOREGROUND
    UNIQUE_NAME = "SET_FMLFE"
    DISPLAY_NAME = "Estimate foreground (FMLFE)"

    def estimate(self, images: torch.Tensor, masks: torch.Tensor, implementation: str,
                 regularization: float, n_small_iterations: int, n_big_iterations: int,
                 small_size: int, gradient_weight: float):
        try:
            foregrounds, backgrounds = fmlfe(
                images=images,
                masks=masks,
                logger=logger,
                implementation=implementation,
                regularization=regularization,
                n_small_iterations=n_small_iterations,
                n_big_iterations=n_big_iterations,
                small_size=small_size,
                gradient_weight=gradient_weight
            )

            return (foregrounds, backgrounds, masks,)

        except Exception as e:
            # This ensures that if all backends fail, the error is clearly visible in the ComfyUI console.
            logger.error("Failed to execute ML Foreground Estimation. All backends failed.")
            logger.error(f"Last error: {e}")
            # Raising the exception will stop the workflow and show the error to the user.
            raise e


class CreateEmptyImage(ComfyNodeABC):
    """
    Create a solid-color image.

    The output dimensions can be specified manually or inherited from an optional input image (`reference`).
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "width": (IO.INT, {
                    "default": 1024,
                    "min": 1,
                    "max": 8192,
                    "step": 8,
                    "tooltip": "The width of the new image in pixels. This value is ignored if a `reference` is provided."
                }),
                "height": (IO.INT, {
                    "default": 1024,
                    "min": 1,
                    "max": 8192,
                    "step": 8,
                    "tooltip": "The height of the new image in pixels. This value is ignored if a `reference` is provided."
                }),
                "batch_size": (IO.INT, {
                    "default": 1,
                    "min": 1,
                    "max": 64,
                    "tooltip": "The number of images to create in the batch. This value is ignored if a `reference` "
                               "is provided."
                }),
                "color": COLOR_OPT,
            },
            "optional": {
                "reference": (IO.IMAGE, {
                    "tooltip": "If an image is connected here, its dimensions (batch size, height, and width) will be "
                               "used for the new image, overriding the manual width, height, and batch_size inputs."
                }),
            }
        }

    RETURN_TYPES = (IO.IMAGE,)
    RETURN_NAMES = ("image",)
    FUNCTION = "create_image"
    CATEGORY = BASE_CATEGORY + "/generation"
    UNIQUE_NAME = "SET_CreateEmptyImage"
    DISPLAY_NAME = "Create Empty Image"

    def create_image(self, width: int, height: int, batch_size: int, color: str,
                     reference: Optional[torch.Tensor] = None):
        # --- 1. Determine the final shape of the output tensor ---
        if reference is not None:
            # If an image is provided, its shape overrides the manual inputs
            b, h, w, _ = reference.shape
        else:
            b, h, w = batch_size, height, width

        # --- 2. Parse the color string ---
        # The function returns a tuple of floats in the [0, 1] range
        rgb_color = color_to_rgb_float(logger, color)

        # --- 3. Create the tensor efficiently ---
        # Create a small color tensor and then expand it to the final size.
        # This is highly memory-efficient as it creates a view, not a full-size copy.
        # Tensors should be created on the CPU by default in generator nodes.
        color_tensor = torch.tensor(rgb_color, dtype=torch.float32, device="cpu").view(1, 1, 1, 3)
        final_image = color_tensor.expand(b, h, w, 3)

        return (final_image,)


# Adapted from KJNodes, credits to Kijai
# - When target_width/target_height are 0 we use the image size
# - Added control over the transparency of the padded area (pad_transparency)
# - Handle RGBA images
class ImagePad(ComfyNodeABC):
    """
    Pad the input image and optionally mask with the specified padding.

    The `target_width`/`target_height` overrides `left`, `right`, `top` and `bottom` values.
    """
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": (IO.IMAGE, ),
                "left": PAD_SIZE_OPT,
                "right": PAD_SIZE_OPT,
                "top": PAD_SIZE_OPT,
                "bottom": PAD_SIZE_OPT,
                "extra_padding": PAD_SIZE_OPT,
                "pad_mode": (["edge", "edge_pixel", "color", "pillarbox_blur"],),
                "color": COLOR_OPT,
            },
            "optional": {
                "mask": (IO.MASK, ),
                "target_width": SIZE_OPT_FI,
                "target_height": SIZE_OPT_FI,
                "pad_transparency": PAD_TRANS,
            }
        }

    RETURN_TYPES = (IO.IMAGE, IO.MASK, )
    RETURN_NAMES = ("images", "masks",)
    FUNCTION = "pad"
    CATEGORY = BASE_CATEGORY + "/" + MANIPULATION_CATEGORY
    UNIQUE_NAME = "SET_ImagePad"
    DISPLAY_NAME = "Pad Image (KJ/SET)"

    def pad(self, image, left, right, top, bottom, extra_padding, color, pad_mode, mask=None, target_width=None,
            target_height=None, pad_transparency=1.0):
        B, H, W, C = image.shape

        # Resize masks to image dimensions if necessary
        if mask is not None:
            BM, HM, WM = mask.shape
            if HM != H or WM != W:
                mask = F.interpolate(mask.unsqueeze(1), size=(H, W), mode=MASK_UPSCALE).squeeze(1)

        # Parse background color
        color_tuple = color_to_rgb_float(logger, color)
        if C == 4 and len(color_tuple) == 3:
            color_tuple += (1.0 - pad_transparency,)  # Use transparent color to pad RGBA images. 0 is transparent for RGBA
        bg_color = torch.tensor(color_tuple, dtype=image.dtype, device=image.device)

        # Calculate padding sizes with extra padding
        if target_width is not None and target_height is not None:
            # SET: If any of them is 0 use the current value
            if target_width == 0:
                target_width = W
            if target_height == 0:
                target_height = H

            if extra_padding > 0:
                image = upscale_comfy(image, W - extra_padding, H - extra_padding, BEST_UPSCALE)
                B, H, W, C = image.shape

            padded_width = target_width
            padded_height = target_height
            pad_left = (padded_width - W) // 2
            pad_right = padded_width - W - pad_left
            pad_top = (padded_height - H) // 2
            pad_bottom = padded_height - H - pad_top
        else:
            pad_left = left + extra_padding
            pad_right = right + extra_padding
            pad_top = top + extra_padding
            pad_bottom = bottom + extra_padding

            padded_width = W + pad_left + pad_right
            padded_height = H + pad_top + pad_bottom

        # Pillarbox blur mode
        if pad_mode == "pillarbox_blur":
            def _gaussian_blur_nchw(img_nchw, sigma_px):
                if sigma_px <= 0:
                    return img_nchw
                radius = max(1, int(3.0 * float(sigma_px)))
                k = 2 * radius + 1
                x = torch.arange(-radius, radius + 1, device=img_nchw.device, dtype=img_nchw.dtype)
                k1 = torch.exp(-(x * x) / (2.0 * float(sigma_px) * float(sigma_px)))
                k1 = k1 / k1.sum()
                kx = k1.view(1, 1, 1, k)
                ky = k1.view(1, 1, k, 1)
                c = img_nchw.shape[1]
                kx = kx.repeat(c, 1, 1, 1)
                ky = ky.repeat(c, 1, 1, 1)
                img_nchw = F.conv2d(img_nchw, kx, padding=(0, radius), groups=c)
                img_nchw = F.conv2d(img_nchw, ky, padding=(radius, 0), groups=c)
                return img_nchw

            out_image = torch.zeros((B, padded_height, padded_width, C), dtype=image.dtype, device=image.device)
            for b in range(B):
                scale_fill = max(padded_width / float(W), padded_height / float(H)) if (W > 0 and H > 0) else 1.0
                bg_w = max(1, int(round(W * scale_fill)))
                bg_h = max(1, int(round(H * scale_fill)))
                src_b = image[b].movedim(-1, 0).unsqueeze(0)
                bg = upscale(src_b, bg_w, bg_h, "bilinear")
                y0 = max(0, (bg_h - padded_height) // 2)
                x0 = max(0, (bg_w - padded_width) // 2)
                y1 = min(bg_h, y0 + padded_height)
                x1 = min(bg_w, x0 + padded_width)
                bg = bg[:, :, y0:y1, x0:x1]
                if bg.shape[2] != padded_height or bg.shape[3] != padded_width:
                    pad_h = padded_height - bg.shape[2]
                    pad_w = padded_width - bg.shape[3]
                    pad_top_fix = max(0, pad_h // 2)
                    pad_bottom_fix = max(0, pad_h - pad_top_fix)
                    pad_left_fix = max(0, pad_w // 2)
                    pad_right_fix = max(0, pad_w - pad_left_fix)
                    bg = F.pad(bg, (pad_left_fix, pad_right_fix, pad_top_fix, pad_bottom_fix), mode="replicate")
                sigma = max(1.0, 0.006 * float(min(padded_height, padded_width)))
                bg = _gaussian_blur_nchw(bg, sigma_px=sigma)
                if C >= 3:
                    r, g, bch = bg[:, 0:1], bg[:, 1:2], bg[:, 2:3]
                    luma = 0.2126 * r + 0.7152 * g + 0.0722 * bch
                    gray = torch.cat([luma, luma, luma], dim=1)
                    desat = 0.20
                    rgb = torch.cat([r, g, bch], dim=1)
                    rgb = rgb * (1.0 - desat) + gray * desat
                    bg[:, 0:3, :, :] = rgb
                dim = 0.35
                bg = torch.clamp(bg * dim, 0.0, 1.0)
                out_image[b] = bg.squeeze(0).movedim(0, -1)
            out_image[:, pad_top:pad_top+H, pad_left:pad_left+W, :] = image
            # Mask handling for pillarbox_blur
            if mask is not None:
                fg_mask = mask
                out_masks = torch.ones((B, padded_height, padded_width), dtype=image.dtype, device=image.device)
                out_masks[:, pad_top:pad_top+H, pad_left:pad_left+W] = fg_mask
            else:
                out_masks = torch.ones((B, padded_height, padded_width), dtype=image.dtype, device=image.device)
                out_masks[:, pad_top:pad_top+H, pad_left:pad_left+W] = 0.0
            return (out_image, out_masks)

        # Standard pad logic (edge/color)
        out_image = torch.zeros((B, padded_height, padded_width, C), dtype=image.dtype, device=image.device)
        for b in range(B):
            if pad_mode == "edge":
                # Pad with edge color (mean)
                top_edge = image[b, 0, :, :]
                bottom_edge = image[b, H-1, :, :]
                left_edge = image[b, :, 0, :]
                right_edge = image[b, :, W-1, :]
                out_image[b, :pad_top, :, :] = top_edge.mean(dim=0)
                out_image[b, pad_top+H:, :, :] = bottom_edge.mean(dim=0)
                out_image[b, :, :pad_left, :] = left_edge.mean(dim=0)
                out_image[b, :, pad_left+W:, :] = right_edge.mean(dim=0)
                out_image[b, pad_top:pad_top+H, pad_left:pad_left+W, :] = image[b]
            elif pad_mode == "edge_pixel":
                # Pad with exact edge pixel values
                for y in range(pad_top):
                    out_image[b, y, pad_left:pad_left+W, :] = image[b, 0, :, :]
                for y in range(pad_top+H, padded_height):
                    out_image[b, y, pad_left:pad_left+W, :] = image[b, H-1, :, :]
                for x in range(pad_left):
                    out_image[b, pad_top:pad_top+H, x, :] = image[b, :, 0, :]
                for x in range(pad_left+W, padded_width):
                    out_image[b, pad_top:pad_top+H, x, :] = image[b, :, W-1, :]
                out_image[b, :pad_top, :pad_left, :] = image[b, 0, 0, :]
                out_image[b, :pad_top, pad_left+W:, :] = image[b, 0, W-1, :]
                out_image[b, pad_top+H:, :pad_left, :] = image[b, H-1, 0, :]
                out_image[b, pad_top+H:, pad_left+W:, :] = image[b, H-1, W-1, :]
                out_image[b, pad_top:pad_top+H, pad_left:pad_left+W, :] = image[b]
            else:
                # Pad with specified background color
                out_image[b, :, :, :] = bg_color.unsqueeze(0).unsqueeze(0)
                out_image[b, pad_top:pad_top+H, pad_left:pad_left+W, :] = image[b]

        # Note: in the mask 1 is transparent and 0 opaque (reverse of RGBA)
        if mask is not None:
            out_masks = torch.nn.functional.pad(
                mask,
                (pad_left, pad_right, pad_top, pad_bottom),
                mode='replicate' if pad_mode == "edge_pixel" else 'constant',
                value=None if pad_mode == "edge_pixel" else pad_transparency,
            )
        else:
            out_masks = torch.full((B, padded_height, padded_width), pad_transparency, dtype=image.dtype,
                                   device=image.device)
            for m in range(B):
                out_masks[m, pad_top:pad_top+H, pad_left:pad_left+W] = 0.0

        return (out_image, out_masks)


# Adapted from KJNodes, credits to Kijai
# Differences:
# - The color is an string that support various formats
# - We can copy the size of a reference image (found in V1, not in V2)
# - Removed misleading code to compute padded size when width and/or height was missing
# - Added control over the transparency of the padded area
class ImageResize(ComfyNodeABC):
    """
    Resizes the image to the specified width and height

    Size can be retrieved from the input (when w=h=0) or a reference image.

    Keep proportions keeps the aspect ratio of the image, by highest dimension.
    """
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": (IO.IMAGE, {"tooltip": "Image to resize"}),
                "width": SIZE_OPT,
                "height": SIZE_OPT,
                "upscale_method": UPSCALE_OPT,
                "keep_proportion": (["stretch", "resize", "pad", "pad_edge", "pad_edge_pixel", "crop", "pillarbox_blur"],
                                    {"default": "stretch",
                                     "tooltip": "`stretch` doesn't keep the aspect ratio\n"
                                                "`pad` adds `pad_color` bars\n"
                                                "`pad_edge` fills using the edge color\n"
                                                "`resize` always keeps aspect, so W and H might change\n"
                                                "`crop` takes a portion of the image"}),
                "pad_color": COLOR_OPT,
                "crop_position": (["center", "top", "bottom", "left", "right"],
                                  {"default": "center", "tooltip": "Also used for `pad`"}),
                "divisible_by": (IO.INT, {"default": 2, "min": 0, "max": 512, "step": 1,
                                          "tooltip": "Force the final size to be divisible by"}),
            },
            "optional": {
                "mask": (IO.MASK, {"tooltip": "Optional mask for the image\nwill be resized"}),
                "device": (["cpu", "gpu"],),
                "get_image_size": (IO.IMAGE, {"tooltip": "Image size to use as reference"}),
                "per_batch": (IO.INT, {
                    "default": 0, "min": 0, "max": MAX_RESOLUTION, "step": 1,
                    "tooltip": "Process images in sub-batches to reduce memory usage. 0 disables sub-batching."}),
                "pad_transparency": PAD_TRANS,
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
            },
        }

    RETURN_TYPES = (IO.IMAGE, IO.INT, IO.INT, IO.MASK,)
    RETURN_NAMES = ("IMAGE", "width", "height", "mask",)
    FUNCTION = "resize"
    CATEGORY = BASE_CATEGORY + "/" + MANIPULATION_CATEGORY
    UNIQUE_NAME = "SET_ImageResize"
    DISPLAY_NAME = "Resize Image (KJ/SET)"

    def resize(self, image, width, height, keep_proportion, upscale_method, divisible_by, pad_color, crop_position,
               unique_id, device="cpu", mask=None, get_image_size=None, per_batch=0, pad_transparency=1.0):
        B, H, W, C = image.shape

        if device == "gpu":
            if upscale_method == "lanczos":
                raise Exception("Lanczos is not supported on the GPU")
            device = get_default_comfy_device()
        else:
            device = torch.device("cpu")

        # Image size from a reference image
        if get_image_size is not None:
            width = get_image_size.shape[2]
            height = get_image_size.shape[1]

        # Copy the size that is 0
        if width == 0:
            width = W
        if height == 0:
            height = H

        pillarbox_blur = keep_proportion == "pillarbox_blur"

        # Initialize padding variables
        pad_left = pad_right = pad_top = pad_bottom = 0

        # Solve the size for the ones that keeps aspect: resize, pad and pad_edge
        if keep_proportion == "resize" or keep_proportion.startswith("pad") or pillarbox_blur:
            # If one of the dimensions is zero, calculate it to maintain the aspect ratio
            ratio = min(width / W, height / H)
            new_width = round(W * ratio)
            new_height = round(H * ratio)

            if keep_proportion.startswith("pad") or pillarbox_blur:
                # Calculate padding based on position
                if crop_position == "center":
                    pad_left = (width - new_width) // 2
                    pad_right = width - new_width - pad_left
                    pad_top = (height - new_height) // 2
                    pad_bottom = height - new_height - pad_top
                elif crop_position == "top":
                    pad_left = (width - new_width) // 2
                    pad_right = width - new_width - pad_left
                    pad_top = 0
                    pad_bottom = height - new_height
                elif crop_position == "bottom":
                    pad_left = (width - new_width) // 2
                    pad_right = width - new_width - pad_left
                    pad_top = height - new_height
                    pad_bottom = 0
                elif crop_position == "left":
                    pad_left = 0
                    pad_right = width - new_width
                    pad_top = (height - new_height) // 2
                    pad_bottom = height - new_height - pad_top
                elif crop_position == "right":
                    pad_left = width - new_width
                    pad_right = 0
                    pad_top = (height - new_height) // 2
                    pad_bottom = height - new_height - pad_top

            width = new_width
            height = new_height

        if divisible_by > 1:
            width = width - (width % divisible_by)
            height = height - (height % divisible_by)

        # Preflight estimate (log-only when batching is active)
        if per_batch != 0 and B > per_batch:
            try:
                bytes_per_elem = image.element_size()  # typically 4 for float32
                est_total_bytes = B * height * width * C * bytes_per_elem
                est_mb = est_total_bytes / (1024 * 1024)
                msg = f"<tr><td>Resize Image</td><td>estimated output ~{est_mb:.2f} MB; batching {per_batch}/{B}</td></tr>"
                if unique_id and PromptServer is not None:
                    try:
                        PromptServer.instance.send_progress_text(msg, unique_id)
                    except Exception:
                        pass
                logger.info(f"estimated output ~{est_mb:.2f} MB; batching {per_batch}/{B}")
            except Exception:
                pass

        def _process_subbatch(in_image, in_mask, pad_left, pad_right, pad_top, pad_bottom):
            # Avoid unnecessary clones; only move if needed
            out_image = in_image if in_image.device == device else in_image.to(device)
            out_mask = None if in_mask is None else (in_mask if in_mask.device == device else in_mask.to(device))

            # Crop logic
            if keep_proportion == "crop":
                old_height = out_image.shape[-3]
                old_width = out_image.shape[-2]
                old_aspect = old_width / old_height
                new_aspect = width / height

                # Calculate dimensions to keep
                if old_aspect > new_aspect:  # Image is wider than target
                    crop_w = round(old_height * new_aspect)
                    crop_h = old_height
                else:  # Image is taller than target
                    crop_w = old_width
                    crop_h = round(old_width / new_aspect)

                # Calculate crop position
                if crop_position == "center":
                    x = (old_width - crop_w) // 2
                    y = (old_height - crop_h) // 2
                elif crop_position == "top":
                    x = (old_width - crop_w) // 2
                    y = 0
                elif crop_position == "bottom":
                    x = (old_width - crop_w) // 2
                    y = old_height - crop_h
                elif crop_position == "left":
                    x = 0
                    y = (old_height - crop_h) // 2
                elif crop_position == "right":
                    x = old_width - crop_w
                    y = (old_height - crop_h) // 2

                # Apply crop
                out_image = out_image.narrow(-2, x, crop_w).narrow(-3, y, crop_h)
                if out_mask is not None:
                    out_mask = out_mask.narrow(-1, x, crop_w).narrow(-2, y, crop_h)

            # Resize the image
            out_image = upscale_comfy(out_image.movedim, width, height, upscale_method)

            if out_mask is not None:
                out_mask = upscale_comfy(out_mask, width, height, upscale_method)

            # Pad logic
            if (keep_proportion.startswith("pad") or pillarbox_blur) and (pad_left > 0 or pad_right > 0 or pad_top > 0
                                                                          or pad_bottom > 0):
                padded_width = width + pad_left + pad_right
                padded_height = height + pad_top + pad_bottom
                if divisible_by > 1:
                    width_remainder = padded_width % divisible_by
                    height_remainder = padded_height % divisible_by
                    if width_remainder > 0:
                        extra_width = divisible_by - width_remainder
                        pad_right += extra_width
                    if height_remainder > 0:
                        extra_height = divisible_by - height_remainder
                        pad_bottom += extra_height

                pad_mode = (
                    "pillarbox_blur" if pillarbox_blur else
                    "edge" if keep_proportion == "pad_edge" else
                    "edge_pixel" if keep_proportion == "pad_edge_pixel" else
                    "color"
                )
                out_image, out_mask = ImagePad.pad(self, out_image, pad_left, pad_right, pad_top, pad_bottom, 0, pad_color,
                                                   pad_mode, mask=out_mask, pad_transparency=pad_transparency)

            return out_image, out_mask

        # If batching disabled (per_batch==0) or batch fits, process whole batch
        if per_batch == 0 or B <= per_batch:
            out_image, out_mask = _process_subbatch(image, mask, pad_left, pad_right, pad_top, pad_bottom)
        else:
            chunks = []
            mask_chunks = [] if mask is not None else None
            total_batches = (B + per_batch - 1) // per_batch
            current_batch = 0
            for start_idx in range(0, B, per_batch):
                current_batch += 1
                end_idx = min(start_idx + per_batch, B)
                sub_img = image[start_idx:end_idx]
                sub_mask = mask[start_idx:end_idx] if mask is not None else None
                sub_out_img, sub_out_mask = _process_subbatch(sub_img, sub_mask, pad_left, pad_right, pad_top, pad_bottom)
                chunks.append(sub_out_img.cpu())
                if mask is not None:
                    mask_chunks.append(sub_out_mask.cpu() if sub_out_mask is not None else None)
                # Per-batch progress update
                send_progress_text(unique_id, f"<tr><td>Resize Image</td><td>batch {current_batch}/{total_batches}"
                                   " · images {end_idx}/{B}</td></tr>")
            out_image = torch.cat(chunks, dim=0)
            if mask is not None and any(m is not None for m in mask_chunks):
                out_mask = torch.cat([m for m in mask_chunks if m is not None], dim=0)
            else:
                out_mask = None

        # Progress UI
        num_elements = out_image.numel()
        element_size = out_image.element_size()
        memory_size_mb = (num_elements * element_size) / (1024 * 1024)
        send_progress_text(unique_id, f"<tr><td>Output: </td><td><b>{out_image.shape[0]}</b> x <b>{out_image.shape[2]}"
                           f"</b> x <b>{out_image.shape[1]} | {memory_size_mb:.2f} MiB</b></td></tr>")

        return (out_image.cpu(), out_image.shape[2], out_image.shape[1],
                out_mask.cpu() if out_mask is not None else
                torch.zeros(64, 64, device=torch.device("cpu"), dtype=torch.float32))


# Adapted from KJNodes, credits to Kijai
# Difference: reference image `get_image_size`
class ResizeMask(ComfyNodeABC):
    """
    Resizes the mask or batch of masks to the specified width and height.
    """
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "mask": (IO.MASK,),
                "width": SIZE_OPT,
                "height": SIZE_OPT,
                "keep_proportions": (IO.BOOLEAN, {"default": False}),
                "upscale_method": UPSCALE_OPT_MASK,
                "crop": (["disabled", "center"],),
            },
            "optional": {
                "get_image_size": (IO.IMAGE, {"tooltip": "Image size to use as reference"}),
            },
        }

    RETURN_TYPES = (IO.MASK, IO.INT, IO.INT,)
    RETURN_NAMES = ("mask", "width", "height",)
    FUNCTION = "resize"
    CATEGORY = BASE_CATEGORY + "/" + MANIPULATION_CATEGORY
    UNIQUE_NAME = "SET_ResizeMask"
    DISPLAY_NAME = "Resize Mask (KJ/SET)"

    def resize(self, mask, width, height, keep_proportions, upscale_method, crop, get_image_size=None):
        # Image size from a reference image
        if get_image_size is not None:
            width = get_image_size.shape[2]
            height = get_image_size.shape[1]

        if keep_proportions:
            _, oh, ow = mask.shape
            width = ow if width == 0 else width
            height = oh if height == 0 else height
            ratio = min(width / ow, height / oh)
            width = round(ow*ratio)
            height = round(oh*ratio)

        out_mask = upscale_comfy(mask, width, height, upscale_method, crop=crop)

        return (out_mask, out_mask.shape[2], out_mask.shape[1],)


# #################################################################################################
# ImageWithTextLabel
# #################################################################################################


def load_font(font_name, font_size):
    """Loads a font from the cache or from file."""
    if (font_name, font_size) in font_cache:
        return font_cache[(font_name, font_size)]

    try:
        font = ImageFont.truetype(font_name, font_size)
    except IOError:
        try:
            # Fallback for Linux
            font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
            if not os.path.exists(font_path):
                # Fallback to a common system font if the specified one isn't found
                font_path = os.path.join("C:", os.sep, "Windows", "Fonts", f"{font_name}.ttf")
            font = ImageFont.truetype(font_path, font_size)
        except IOError:
            # If all else fails, use the default Pillow font
            logger.warning(f"Font '{font_name}' not found. Falling back to default font.")
            font = ImageFont.load_default()

    font_cache[(font_name, font_size)] = font
    return font


class ImageWithTextLabel(ComfyNodeABC):
    """ Adds a text label to an image """
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "text": (IO.STRING, {
                    "multiline": True, "default": "Your text here",
                    "tooltip": "Label for this image"}),
                "side": (["top", "bottom", "left", "right"],),
                "label_size": (IO.STRING, {
                    "default": "10%",
                    "tooltip": "Expressed as a percentage (i.e. 10%) or absolute number of pixels"}),
                "separation": (IO.STRING, {
                    "default": "1%",
                    "tooltip": "Expressed as a percentage (i.e. 1%) or absolute number of pixels"}),
                "background_color": (IO.STRING, {"default": "white"}),
                "foreground_color": (IO.STRING, {"default": "black"}),
                "font_name": (IO.STRING, {"default": "Arial"}),
            },
            "optional": {
                "image": (IO.IMAGE, {
                    "tooltip": "Image, leave unconnected when using a mask"}),
                "mask": (IO.MASK, {
                    "tooltip": "Mask to be used as image, leave unconnected when using an image"}),
            }
        }

    RETURN_TYPES = (IO.IMAGE,)
    FUNCTION = "add_label"
    CATEGORY = BASE_CATEGORY + "/" + MANIPULATION_CATEGORY
    UNIQUE_NAME = "SET_ImageWithTextLabel"
    DISPLAY_NAME = "Image with text label, good for comparison grids"

    def add_label(self, text: str, side: str, label_size: str, separation: int, background_color: str,
                  foreground_color: str, font_name: str, image: Optional[torch.Tensor] = None,
                  mask: Optional[torch.Tensor] = None):
        if image is None and mask is None:
            raise ValueError("You must provide an image or a mask")
        image = image if image is not None else mask
        # Process multiple images in the batch
        pil_images = [tensor_to_pil(img) for img in image]
        output_images = []

        for pil_img in pil_images:
            original_width, original_height = pil_img.size

            # Parse label size
            if side in ["top", "bottom"]:
                label_dim = parse_size(label_size, original_height)
                separation_px = parse_size(separation, original_height)
            else:  # left, right
                label_dim = parse_size(label_size, original_width)
                separation_px = parse_size(separation, original_width)

            # Calculate new canvas dimensions
            new_width = original_width + 2 * separation_px
            new_height = original_height + 2 * separation_px
            if side in ["top", "bottom"]:
                new_height += label_dim
            else:  # left, right
                new_width += label_dim

            # Create new image canvas
            bg_color_rgb = color_to_rgb_uint8(logger, background_color)
            canvas = Image.new("RGB", (new_width, new_height), bg_color_rgb)

            # Paste the original image with separation
            img_paste_position = (separation_px, separation_px)
            if side == "top":
                img_paste_position = (separation_px, separation_px + label_dim)
            elif side == "left":
                img_paste_position = (separation_px + label_dim, separation_px)

            canvas.paste(pil_img, img_paste_position)
            draw = ImageDraw.Draw(canvas)

            # Define the text area
            text_area = (0, 0, 0, 0)
            if side == "top":
                text_area = (separation_px, separation_px, new_width - separation_px, separation_px + label_dim)
            elif side == "bottom":
                text_area = (separation_px, original_height + separation_px, new_width - separation_px, new_height -
                             separation_px)
            elif side == "left":
                text_area = (separation_px, separation_px, separation_px + label_dim, new_height - separation_px)
            elif side == "right":
                text_area = (original_width + separation_px, separation_px, new_width - separation_px, new_height -
                             separation_px)

            text_area_width = text_area[2] - text_area[0]
            text_area_height = text_area[3] - text_area[1]

            fg_color_rgb = color_to_rgb_uint8(logger, foreground_color)
            text_x = text_area[0] + text_area_width / 2
            text_y = text_area[1] + text_area_height / 2
            font = None

            # --- Find best font size and render text ---
            if side in ["top", "bottom"]:
                # --- HORIZONTAL TEXT LOGIC (with wrapping) ---
                max_font_size = text_area_height
                font_size = max_font_size
                wrapped_text = text

                while font_size > 5:
                    font = load_font(font_name, font_size)

                    # Simple word wrapping
                    lines = []
                    words = text.split()
                    current_line = ""
                    for word in words:
                        if font.getlength(current_line + word + " ") < text_area_width:
                            current_line += word + " "
                        else:
                            lines.append(current_line.strip())
                            current_line = word + " "
                    lines.append(current_line.strip())
                    wrapped_text = "\n".join(lines)

                    # Check if wrapped text fits vertically
                    bbox = draw.multiline_textbbox((0, 0), wrapped_text, font=font)
                    text_height = bbox[3] - bbox[1]

                    if text_height <= text_area_height:
                        break  # Font size is good

                    font_size -= 1
                else:
                    logger.warning("Text could not fit into the designated area even at the smallest font size.")

                draw.multiline_text(
                    (text_x, text_y),
                    wrapped_text,
                    fill=fg_color_rgb,
                    font=font,
                    anchor="mm",  # middle-middle anchor
                    align="center"
                )

            else:  # --- VERTICAL TEXT LOGIC ---
                max_font_size = text_area_width  # For vertical, font size is constrained by width
                font_size = max_font_size

                while font_size > 5:
                    font = load_font(font_name, font_size)
                    # Use textbbox with direction to measure final size
                    bbox = draw.textbbox((0, 0), text, font=font, direction="ttb")
                    text_width = bbox[2] - bbox[0]
                    text_height = bbox[3] - bbox[1]

                    # Check if the vertically rendered text fits in the area
                    if text_width <= text_area_width and text_height <= text_area_height:
                        break  # Font size is good

                    font_size -= 1
                else:
                    logger.warning("Text could not fit into the designated area even at the smallest font size.")

                draw.text(
                    (text_x, text_y),
                    text,
                    fill=fg_color_rgb,
                    font=font,
                    anchor="mm",  # middle-middle anchor
                    direction="ttb"  # Top-to-bottom direction
                )

            output_images.append(pil_to_tensor(canvas).unsqueeze(0))

        # Stack the processed images back into a single tensor
        return (torch.cat(output_images, dim=0),)
