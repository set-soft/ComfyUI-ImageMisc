import numpy as np
import os
from PIL import Image, ImageOps, ImageSequence
from PIL import ImageFile, UnidentifiedImageError
import torch
from . import main_logger

try:
    # We need to import the built-in LoadImage class for ImageDownload
    from nodes import LoadImage
    from folder_paths import get_input_directory
    has_load_image = hasattr(LoadImage, "load_image")
except Exception:
    has_load_image = False

logger = main_logger


def pillow(fn, arg):
    prev_value = None
    try:
        x = fn(arg)
    except (OSError, UnidentifiedImageError, ValueError):  # PIL issues #4472 and #2445, also fixes ComfyUI issue #3416
        prev_value = ImageFile.LOAD_TRUNCATED_IMAGES
        ImageFile.LOAD_TRUNCATED_IMAGES = True
        x = fn(arg)
    finally:
        if prev_value is not None:
            ImageFile.LOAD_TRUNCATED_IMAGES = prev_value
    return x


class CustomLoadImage(object):
    def load_image(self, image):
        """ ComfyUI 0.3.59 loader """
        image_path = image

        img = pillow(Image.open, image_path)

        output_images = []
        output_masks = []
        w, h = None, None

        excluded_formats = ['MPO']

        for i in ImageSequence.Iterator(img):
            i = pillow(ImageOps.exif_transpose, i)

            if i.mode == 'I':
                i = i.point(lambda i: i * (1 / 255))
            image = i.convert("RGB")

            if len(output_images) == 0:
                w = image.size[0]
                h = image.size[1]

            if image.size[0] != w or image.size[1] != h:
                continue

            image = np.array(image).astype(np.float32) / 255.0
            image = torch.from_numpy(image)[None,]
            if 'A' in i.getbands():
                mask = np.array(i.getchannel('A')).astype(np.float32) / 255.0
                mask = 1. - torch.from_numpy(mask)
            elif i.mode == 'P' and 'transparency' in i.info:
                mask = np.array(i.convert('RGBA').getchannel('A')).astype(np.float32) / 255.0
                mask = 1. - torch.from_numpy(mask)
            else:
                mask = torch.zeros((64, 64), dtype=torch.float32, device="cpu")
            output_images.append(image)
            output_masks.append(mask.unsqueeze(0))

        if len(output_images) > 1 and img.format not in excluded_formats:
            output_image = torch.cat(output_images, dim=0)
            output_mask = torch.cat(output_masks, dim=0)
        else:
            output_image = output_images[0]
            output_mask = output_masks[0]

        return (output_image, output_mask)


def load_image_wrapper(file_name, embed_transparency, disp_name=None):
    disp_name = disp_name or file_name

    # --- REUSE ComfyUI's LoadImage LOGIC ---
    try:
        if has_load_image:
            # Instantiate the built-in LoadImage node
            loader_instance = LoadImage()

            # The LoadImage node's `load_image` method expects the filename as passed
            # by the ComfyUI widget, which is just the filename. It internally
            # resolves the path using folder_paths.

            logger.debug(f"Calling built-in LoadImage.load_image() with filename: '{file_name}'")
        else:
            # Instantiate the built-in LoadImage node
            loader_instance = CustomLoadImage()
            logger.debug(f"Calling our CustomLoadImage.load_image() with filename: '{file_name}'")

        # Call the method and return its result directly
        result = loader_instance.load_image(file_name)
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
        if os.path.isabs(file_name):
            ff_name = os.path.relpath(file_name, get_input_directory())
            fname = os.path.basename(ff_name)
            dname = os.path.dirname(ff_name)
        else:
            fname = file_name
            dname = ""
        downloaded_file = {
             "images": [{
                 "filename": fname,
                 "subfolder": dname,
                 "type": "input"  # We stored the file in the "input" folder
             }]
        }
        return {"ui": downloaded_file, "result": result}

    except Exception as e:
        logger.error(f"Failed to load image '{disp_name}' using built-in LoadImage node: {e}", exc_info=True)
        # Re-raise to make the error visible in ComfyUI
        raise IOError(f"Could not load the image file '{disp_name}' using the standard loader. "
                      "It may be corrupt or in an unsupported format.") from e
