# -*- coding: utf-8 -*-
# Copyright (c) 2025 Salvador E. Tropea
# Copyright (c) 2025 Instituto Nacional de Tecnologïa Industrial
# License: GPL-3.0
#
# Project: ComfyUI-ImageMisc
# Credits:
# - ImagePad, ImageResize and ResizeMask are from Kijai (https://github.com/kijai/ComfyUI-KJNodes/) v1.1.7
# - Assisted by Gemini 2.5 Pro
from copy import deepcopy
import numpy as np
import os
from PIL import Image  # Import the Python Imaging Library
from seconohe.apply_mask import apply_mask
from seconohe.foreground_estimation.affce import affce
from seconohe.foreground_estimation.fmlfe import fmlfe, IMPL_PRIORITY
from seconohe.downloader import download_file
from seconohe.color import color_to_rgb_float
# We are the main source, so we use the main_logger
from . import main_logger
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from typing import Optional
try:
    from folder_paths import get_input_directory   # To get the ComfyUI input directory
    from comfy import model_management
    from comfy.utils import common_upscale
except ModuleNotFoundError:
    # No ComfyUI, this is a test environment
    def get_input_directory():
        return ""

try:
    from nodes import ImageScale
except Exception:
    class ImageScale(object):
        upscale_methods = ["nearest-exact", "bilinear", "area", "bicubic", "lanczos"]
try:
    from nodes import MAX_RESOLUTION
except Exception:
    MAX_RESOLUTION = 16384
try:
    from server import PromptServer
except ModuleNotFoundError:
    PromptServer = None
try:
    # We need to import the built-in LoadImage class for ImageDownload
    from nodes import LoadImage
    has_load_image = True
except Exception:
    has_load_image = False

logger = main_logger
BASE_CATEGORY = "image"
IO_CATEGORY = "io"
MANIPULATION_CATEGORY = "manipulation"
NORMALIZATION = "normalization"
FOREGROUND = "foreground"
BLUR_SIZE_OPT = ("INT", {"default": 90, "min": 1, "max": 255, "step": 1, })
BLUR_SIZE_TWO_OPT = ("INT", {"default": 6, "min": 1, "max": 255, "step": 1, })
COLOR_OPT = ("STRING", {
                "default": "#000000",
                "tooltip": "Color for fill.\n"
                           "Can be an hexadecimal (#RRGGBB).\n"
                           "Can comma separated RGB values in [0-255] or [0-1.0] range."})
DEFAULT_UPSCALE = 'bicubic'     # transforms.InterpolationMode.BICUBIC.value
MASK_UPSCALE = 'nearest-exact'  # transforms.InterpolationMode.NEAREST_EXACT.value
BEST_UPSCALE = 'lanczos'        # transforms.InterpolationMode.LANCZOS.value
UPSCALE_OPT = (ImageScale.upscale_methods, {  # [mode.value for mode in transforms.InterpolationMode]
                "default": DEFAULT_UPSCALE,
                "tooltip": "Interpolation method for image resize"
                })
UPSCALE_OPT_MASK = deepcopy(UPSCALE_OPT)
UPSCALE_OPT_MASK[1]["default"] = MASK_UPSCALE
PAD_SIZE_OPT = ("INT", {"default": 0, "min": 0, "max": MAX_RESOLUTION, "step": 1, })
SIZE_OPT = ("INT", {"default": 512, "min": 0, "max": MAX_RESOLUTION, "step": 1})
SIZE_OPT_FI = deepcopy(SIZE_OPT)
SIZE_OPT_FI[1]["forceInput"] = True
SIZE_OPT_FI[1]["tooltip"] = ("Connect both `target` inputs\n"
                             "If 0 the size of the image is used\n"
                             "Overrides left/right/top/bottom")
SIZE_OPT[1]["tooltip"] = "Used when no `get_image_size` is provided"
PAD_TRANS = ("FLOAT", {
                "default": 1.0,
                "min": 0.0,
                "max": 1.0,
                "step": 0.1,
                "display": "number",
                "tooltip": ("The transparency for the padded area for all modes except `edge_pixel`."
                            "1.0 is fully transparent, 0.0 is fully opaque.")})
NORM_PARAM = ("FLOAT", {
                "default": 1.0,
                "min": 0.0,
                "max": 1.0,
                "step": 0.1,
                "display": "number"})


def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    """Converts a single image tensor (H, W, C) [0, 1] to a Pillow Image."""
    np_image = (tensor.cpu().numpy() * 255).astype(np.uint8)
    return Image.fromarray(np_image)


def pil_to_tensor(pil_image: Image.Image) -> torch.Tensor:
    """Converts a Pillow Image to a tensor (H, W, C) [0, 1]."""
    np_image = np.array(pil_image).astype(np.float32) / 255.0
    return torch.from_numpy(np_image)


def upscale(image, width, height, upscale_method):
    # return F.interpolate(image, size=(height, width), mode=upscale_method)
    return common_upscale(image, width, height, upscale_method, crop="disabled")


if has_load_image:
    class ImageDownload:
        @classmethod
        def INPUT_TYPES(cls):
            return {
                "required": {
                    "base_url": ("STRING", {
                        "default":
                            "https://raw.githubusercontent.com/set-soft/AudioSeparation/refs/heads/main/example_workflows/",
                        "tooltip": "The base URL where the image file is located."
                    }),
                    "filename": ("STRING", {
                        "default": "audioseparation_logo.jpg",
                        "tooltip": "The name of the image file to download (e.g., photo.jpg, art.png)."
                    }),
                },
                "optional": {
                    "image_bypass": ("IMAGE", {
                         "tooltip": "If this image is present will be used instead of the downloaded one"
                    }),
                    "mask_bypass": ("MASK", {"tooltip": "If this mask is present will be used instead of the downloaded one"}),
                    "local_name": ("STRING", {
                        "default": "",
                        "tooltip": "The name used locally. Leave empty to use `filename`"
                    }),
                    "embed_transparency": ("BOOLEAN", {
                        "default": False,
                        "tooltip": "Create RGBA images when they have transparency."
                    }),
                }
            }

        RETURN_TYPES = ("IMAGE", "MASK")
        RETURN_NAMES = ("image", "alpha_mask")
        FUNCTION = "load_or_download_image"
        CATEGORY = BASE_CATEGORY + "/" + IO_CATEGORY
        DESCRIPTION = ("Downloads an image to ComfyUI's 'input' directory if it doesn't exist, then loads it using the "
                       "built-in LoadImage logic.")
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

            # --- REUSE ComfyUI's LoadImage LOGIC ---
            try:
                # Instantiate the built-in LoadImage node
                loader_instance = LoadImage()

                # The LoadImage node's `load_image` method expects the filename as passed
                # by the ComfyUI widget, which is just the filename. It internally
                # resolves the path using folder_paths.

                logger.debug(f"Calling built-in LoadImage.load_image() with filename: '{dest_fname}'")

                # Call the method and return its result directly
                result = loader_instance.load_image(dest_fname)
                # Create an RGBA image if needed
                if embed_transparency:
                    image, mask = result
                    # Expand the mask to (b, h, w, 1)
                    mask = mask[..., None]
                    # Concatenate image and mask into (b, h, w, 4)
                    image_with_alpha = torch.cat([image, 1.0 - mask], dim=-1)
                    result = (image_with_alpha, mask)
                # This information is for the preview, as we are an output node and we return images
                # they will be displayed in our node. Quite simple.
                downloaded_file = {
                     "images": [{
                         "filename": dest_fname,
                         "subfolder": "",
                         "type": "input"  # We stored the file in the "input" folder
                     }]
                }
                return {"ui": downloaded_file, "result": result}

            except Exception as e:
                logger.error(f"Failed to load image '{filename}' using built-in LoadImage node: {e}", exc_info=True)
                # Re-raise to make the error visible in ComfyUI
                raise IOError(f"Could not load the image file '{filename}' using the standard loader. "
                              "It may be corrupt or in an unsupported format.") from e
else:
    logger.error("Failed to import ComfyUI `LoadImage`, please fill an issue here: "
                 "https://github.com/set-soft/ComfyUI-ImageMisc/issues")


class CompositeFace:
    """
    A ComfyUI node to composite (paste) animated face crops back onto reference images.
    It handles a M-to-N relationship, where M reference images and bboxes correspond
    to M*N animated face images.
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "animated": ("IMAGE",),      # The M*N batch of cropped faces
                "reference": ("IMAGE",),     # The M batch of original context images
                "bboxes": ("BBOX",),         # The M list of (x, y, w, h) tuples
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "composite"

    CATEGORY = BASE_CATEGORY + "/" + MANIPULATION_CATEGORY
    DESCRIPTION = ("Inserts the `animated` face in the `reference` images at the `bboxes` coordinates.")
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
    A ComfyUI node to composite animated frames onto reference frames on a 1-to-1 basis.
    It expects the 'animated' and 'reference' batches to have the same number of frames.
    It uses the *first* bounding box from the 'bboxes' input for all frames.
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "animated": ("IMAGE",),      # The batch of cropped/processed frames
                "reference": ("IMAGE",),     # The batch of original frames
                "bboxes": ("BBOX",),         # A list of bboxes; only the first is used
            },
        }

    CATEGORY = BASE_CATEGORY + "/" + MANIPULATION_CATEGORY
    DESCRIPTION = ("Inserts the `animated` face in the `reference` video at the `bboxes` coordinates.")
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


class NormalizeToImageNetDataset():
    """
    A ComfyUI node to normalize the values to the mean/std of the ImageNet dataset
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
            },
        }
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "normalize"
    CATEGORY = BASE_CATEGORY + "/" + NORMALIZATION
    DESCRIPTION = ("Normalize the image to the ImageNet dataset")
    UNIQUE_NAME = "SET_NormalizeToImageNetDataset"
    DISPLAY_NAME = "Normalize Image to ImageNet"

    def normalize(self, image: torch.Tensor):
        return (TF.normalize(image.permute(0, 3, 1, 2),  # BHWC -> BCHW
                             mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]).permute(0, 2, 3, 1),)  # BCHW -> BHWC


class NormalizeToRangeMinus05to05():
    """
    A ComfyUI node to normalize the values to the [-0.5, 0.5] range
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"image": ("IMAGE",), }, }
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "normalize"
    CATEGORY = BASE_CATEGORY + "/" + NORMALIZATION
    DESCRIPTION = ("Normalize the image to [-0.5, 0.5]")
    UNIQUE_NAME = "SET_NormalizeToRangeMinus05to05"
    DISPLAY_NAME = "Normalize Image to [-0.5, 0.5]"

    def normalize(self, image: torch.Tensor):
        return (TF.normalize(image.permute(0, 3, 1, 2),  # BHWC -> BCHW
                             mean=[0.5, 0.5, 0.5],
                             std=[1.0, 1.0, 1.0]).permute(0, 2, 3, 1),)  # BCHW -> BHWC


class NormalizeToRangeMinus1to1():
    """
    A ComfyUI node to normalize the values to the [-1, 1] range
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"image": ("IMAGE",), }, }
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "normalize"
    CATEGORY = BASE_CATEGORY + "/" + NORMALIZATION
    DESCRIPTION = ("Normalize the image to [-1, 1]")
    UNIQUE_NAME = "SET_NormalizeToRangeMinus1to1"
    DISPLAY_NAME = "Normalize Image to [-1, 1] (i.e. GAN)"

    def normalize(self, image: torch.Tensor):
        return (TF.normalize(image.permute(0, 3, 1, 2),  # BHWC -> BCHW
                             mean=[0.5, 0.5, 0.5],
                             std=[0.5, 0.5, 0.5]).permute(0, 2, 3, 1),)  # BCHW -> BHWC


class NormalizeArbitrary():
    """
    A ComfyUI node to normalize the values to arbitrary mean/std
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "parameters": ("NORM_PARAMS",),
            },
        }
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "normalize"
    CATEGORY = BASE_CATEGORY + "/" + NORMALIZATION
    DESCRIPTION = ("Normalize the image to the provided parameters")
    UNIQUE_NAME = "SET_NormalizeArbitrary"
    DISPLAY_NAME = "Arbitrary Normalize"

    def normalize(self, image: torch.Tensor, parameters):
        return (TF.normalize(image.movedim(-1, 1),  # BHWC -> BCHW
                             mean=parameters["mean"],
                             std=parameters["std"]).movedim(1, -1),)  # BCHW -> BHWC


class NormalizeParameters():
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
    DESCRIPTION = ("Parameters for the arbitrary normalization")
    UNIQUE_NAME = "SET_NormalizeParameters"
    DISPLAY_NAME = "Normalize Parameters"

    def normalize(self, mean_red, mean_green, mean_blue, std_red, std_green, std_blue):
        return ({"mean": [mean_red, mean_green, mean_blue], "std": [std_red, std_green, std_blue]},)


class ApplyMaskAFFCE:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "masks": ("MASK",),
                "blur_size": BLUR_SIZE_OPT,
                "blur_size_two": BLUR_SIZE_TWO_OPT,
                "fill_color": ("BOOLEAN", {
                    "default": False,
                    "tooltip": ("Fill the background using a color.\n"
                                "Returns an RGB image, otherwise an RGBA.")
                }),
                "color": COLOR_OPT,
                "batched":  ("BOOLEAN", {
                    "default": True,
                    "tooltip": ("Process the images at once.\n"
                                "Faster, needs more memory")
                }),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK",)
    RETURN_NAMES = ("image", "mask",)
    FUNCTION = "get_foreground"
    CATEGORY = BASE_CATEGORY + "/" + MANIPULATION_CATEGORY
    DESCRIPTION = ("Apply a mask to an image using\n"
                   "Approximate Fast Foreground Colour Estimation.\n"
                   "https://github.com/Photoroom/fast-foreground-estimation")
    UNIQUE_NAME = "SET_ApplyMaskAFFCE"
    DISPLAY_NAME = "Apply Mask using AFFCE"

    def get_foreground(self, images, masks, blur_size=91, blur_size_two=7, fill_color=False, color=None, batched=True):
        out_images = apply_mask(logger, images, masks, model_management.get_torch_device(), blur_size, blur_size_two,
                                fill_color, color, batched)
        return out_images.cpu(), masks.cpu()


class AFFCE:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "masks": ("MASK",),
                "blur_size": BLUR_SIZE_OPT,
                "blur_size_two": BLUR_SIZE_TWO_OPT,
                "batched":  ("BOOLEAN", {
                    "default": True,
                    "tooltip": ("Process the images at once.\n"
                                "Faster, needs more memory")
                }),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK",)
    RETURN_NAMES = ("foreground", "mask",)
    FUNCTION = "get_foreground"
    CATEGORY = BASE_CATEGORY + "/" + FOREGROUND
    DESCRIPTION = ("Estimate the foreground image using\n"
                   "Approximate Fast Foreground Colour Estimation.\n"
                   "https://github.com/Photoroom/fast-foreground-estimation")
    UNIQUE_NAME = "SET_AFFCE"
    DISPLAY_NAME = "Estimate foreground (AFFCE)"

    def get_foreground(self, images, masks, blur_size=91, blur_size_two=7, batched=True):
        device = model_management.get_torch_device()
        images_on_device = images.to(device)
        masks_on_device = masks.to(device)

        out_images = affce(images_on_device, masks_on_device, r1=blur_size, r2=blur_size_two, batched=batched)

        return out_images.cpu(), masks.cpu()


class FMLFE:
    """
    A ComfyUI node that uses the Fast Multi-Level Foreground Estimation algorithm
    to produce a high-quality foreground and background separation. It can
    intelligently select the best available backend (CuPy, OpenCL, Numba, or PyTorch).
    """

    @classmethod
    def INPUT_TYPES(cls):
        # Create the dropdown list for the implementation choice
        impl_list = ['auto'] + IMPL_PRIORITY

        return {
            "required": {
                "images": ("IMAGE", {
                    "tooltip": "The source image(s) from which to estimate the foreground and background."
                }),
                "masks": ("MASK", {
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
                "regularization": ("FLOAT", {
                    "default": 1e-5,
                    "min": 0.0,
                    "max": 0.1,
                    "step": 1e-5,
                    "display": "number",
                    "tooltip": "The regularization strength (epsilon). This acts as a smoothness prior. "
                               "Higher values result in smoother, more blended foreground and background colors, "
                               "but may lose very fine details. Lower values preserve more detail but can be noisier."
                }),
                "n_small_iterations": ("INT", {
                    "default": 10,
                    "min": 1,
                    "max": 100,
                    "tooltip": "The number of solver iterations to perform on the lower-resolution levels of the "
                               "image pyramid. More iterations can improve quality at the cost of speed."
                }),
                "n_big_iterations": ("INT", {
                    "default": 2,
                    "min": 1,
                    "max": 100,
                    "tooltip": "The number of solver iterations to perform on the higher-resolution (larger) levels "
                               "of the image pyramid. Fewer iterations are typically needed at high resolution as the "
                               "details are propagated up from the smaller levels."
                }),
                "small_size": ("INT", {
                    "default": 32,
                    "min": 8,
                    "max": 256,
                    "tooltip": "The pixel dimension threshold. Image pyramid levels smaller than this size will use "
                               "the higher 'n_small_iterations' count, while larger levels will use 'n_big_iterations'."
                }),
                "gradient_weight": ("FLOAT", {
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

    RETURN_TYPES = ("IMAGE", "IMAGE", "MASK",)
    RETURN_NAMES = ("foreground", "background", "mask")
    FUNCTION = "estimate"
    CATEGORY = BASE_CATEGORY + "/" + FOREGROUND
    DESCRIPTION = ("Estimate the foreground image using\n"
                   "Fast Multi-Level Foreground Estimation.")
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


class CreateEmptyImage:
    """
    A ComfyUI node to create a solid-color image tensor.
    The output dimensions can be specified manually or inherited from an optional input image.
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "width": ("INT", {
                    "default": 1024,
                    "min": 1,
                    "max": 8192,
                    "step": 8,
                    "tooltip": "The width of the new image in pixels. This value is ignored if a `reference` is provided."
                }),
                "height": ("INT", {
                    "default": 1024,
                    "min": 1,
                    "max": 8192,
                    "step": 8,
                    "tooltip": "The height of the new image in pixels. This value is ignored if a `reference` is provided."
                }),
                "batch_size": ("INT", {
                    "default": 1,
                    "min": 1,
                    "max": 64,
                    "tooltip": "The number of images to create in the batch. This value is ignored if a `reference` "
                               "is provided."
                }),
                "color": COLOR_OPT,
            },
            "optional": {
                "reference": ("IMAGE", {
                    "tooltip": "If an image is connected here, its dimensions (batch size, height, and width) will be "
                               "used for the new image, overriding the manual width, height, and batch_size inputs."
                }),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "create_image"
    CATEGORY = BASE_CATEGORY + "/generation"
    DESCRIPTION = ("Create a solid-color image.\n"
                   "If the optional image is provides uses its shape.")
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
class ImagePad:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE", ),
                "left": PAD_SIZE_OPT,
                "right": PAD_SIZE_OPT,
                "top": PAD_SIZE_OPT,
                "bottom": PAD_SIZE_OPT,
                "extra_padding": PAD_SIZE_OPT,
                "pad_mode": (["edge", "edge_pixel", "color", "pillarbox_blur"],),
                "color": COLOR_OPT,
            },
            "optional": {
                "mask": ("MASK", ),
                "target_width": SIZE_OPT_FI,
                "target_height": SIZE_OPT_FI,
                "pad_transparency": PAD_TRANS,
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK", )
    RETURN_NAMES = ("images", "masks",)
    FUNCTION = "pad"
    CATEGORY = BASE_CATEGORY + "/" + MANIPULATION_CATEGORY
    DESCRIPTION = ("Pad the input image and optionally mask with the specified padding.\n"
                   "The `target_width`/`target_height` overrides left, right, top and bottom.")
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
                image = upscale(image.movedim(-1, 1), W - extra_padding, H - extra_padding, BEST_UPSCALE).movedim(1, -1)
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
class ImageResize:
    """
    A resize and crop node, from ImageResizeKJv2
    """
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE", {"tooltip": "Image to resize"}),
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
                "divisible_by": ("INT", {"default": 2, "min": 0, "max": 512, "step": 1,
                                         "tooltip": "Force the final size to be divisible by"}),
            },
            "optional": {
                "mask": ("MASK", {"tooltip": "Optional mask for the image\nwill be resized"}),
                "device": (["cpu", "gpu"],),
                "get_image_size": ("IMAGE", {"tooltip": "Image size to use as reference"}),
                "per_batch": ("INT", {
                    "default": 0, "min": 0, "max": MAX_RESOLUTION, "step": 1,
                    "tooltip": "Process images in sub-batches to reduce memory usage. 0 disables sub-batching."}),
                "pad_transparency": PAD_TRANS,
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
            },
        }

    RETURN_TYPES = ("IMAGE", "INT", "INT", "MASK",)
    RETURN_NAMES = ("IMAGE", "width", "height", "mask",)
    FUNCTION = "resize"
    CATEGORY = BASE_CATEGORY + "/" + MANIPULATION_CATEGORY
    DESCRIPTION = ("Resizes the image to the specified width and height.\n"
                   "Size can be retrieved from the input (when w=h=0) or a reference image.\n\n"
                   "Keep proportions keeps the aspect ratio of the image, by\n"
                   "highest dimension.")
    UNIQUE_NAME = "SET_ImageResize"
    DISPLAY_NAME = "Resize Image (KJ/SET)"

    def resize(self, image, width, height, keep_proportion, upscale_method, divisible_by, pad_color, crop_position,
               unique_id, device="cpu", mask=None, get_image_size=None, per_batch=0, pad_transparency=1.0):
        B, H, W, C = image.shape

        if device == "gpu":
            if upscale_method == "lanczos":
                raise Exception("Lanczos is not supported on the GPU")
            device = model_management.get_torch_device()
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
            out_image = upscale(out_image.movedim(-1, 1), width, height, upscale_method).movedim(1, -1)

            if out_mask is not None:
                if upscale_method == "lanczos":
                    out_mask = upscale(out_mask.unsqueeze(1).repeat(1, 3, 1, 1), width, height,
                                       upscale_method).movedim(1, -1)[:, :, :, 0]
                else:
                    out_mask = upscale(out_mask.unsqueeze(1), width, height, upscale_method).squeeze(1)

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
                if unique_id and PromptServer is not None:
                    try:
                        PromptServer.instance.send_progress_text(
                            f"<tr><td>Resize Image</td><td>batch {current_batch}/{total_batches} · images {end_idx}/{B}"
                            "</td></tr>",
                            unique_id
                        )
                    except Exception:
                        pass
                else:
                    try:
                        logger.info(f"batch {current_batch}/{total_batches} · images {end_idx}/{B}")
                    except Exception:
                        pass
            out_image = torch.cat(chunks, dim=0)
            if mask is not None and any(m is not None for m in mask_chunks):
                out_mask = torch.cat([m for m in mask_chunks if m is not None], dim=0)
            else:
                out_mask = None

        # Progress UI
        if unique_id and PromptServer is not None:
            try:
                num_elements = out_image.numel()
                element_size = out_image.element_size()
                memory_size_mb = (num_elements * element_size) / (1024 * 1024)

                PromptServer.instance.send_progress_text(
                    f"<tr><td>Output: </td><td><b>{out_image.shape[0]}</b> x <b>{out_image.shape[2]}</b> x <b>"
                    f"{out_image.shape[1]} | {memory_size_mb:.2f}MB</b></td></tr>",
                    unique_id
                )
            except Exception:
                pass

        return (out_image.cpu(), out_image.shape[2], out_image.shape[1],
                out_mask.cpu() if out_mask is not None else
                torch.zeros(64, 64, device=torch.device("cpu"), dtype=torch.float32))


# Adapted from KJNodes, credits to Kijai
# Difference: reference image `get_image_size`
class ResizeMask:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "mask": ("MASK",),
                "width": SIZE_OPT,
                "height": SIZE_OPT,
                "keep_proportions": ("BOOLEAN", {"default": False}),
                "upscale_method": UPSCALE_OPT_MASK,
                "crop": (["disabled", "center"],),
            },
            "optional": {
                "get_image_size": ("IMAGE", {"tooltip": "Image size to use as reference"}),
            },
        }

    RETURN_TYPES = ("MASK", "INT", "INT",)
    RETURN_NAMES = ("mask", "width", "height",)
    FUNCTION = "resize"
    CATEGORY = BASE_CATEGORY + "/" + MANIPULATION_CATEGORY
    DESCRIPTION = "Resizes the mask or batch of masks to the specified width and height."
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

        if upscale_method == "lanczos":
            out_mask = common_upscale(mask.unsqueeze(1).repeat(1, 3, 1, 1), width, height, upscale_method,
                                      crop=crop).movedim(1, -1)[:, :, :, 0]
        else:
            out_mask = common_upscale(mask.unsqueeze(1), width, height, upscale_method, crop=crop).squeeze(1)

        return (out_mask, out_mask.shape[2], out_mask.shape[1],)
