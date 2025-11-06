import json
import numpy as np
import os
from PIL import Image, ImageOps, ImageSequence, ImageFile, UnidentifiedImageError
from PIL.PngImagePlugin import PngInfo
import torch
from . import main_logger

try:
    # We need to import the built-in LoadImage class for ImageDownload
    from nodes import LoadImage
    from folder_paths import get_input_directory, get_output_directory
    has_load_image = hasattr(LoadImage, "load_image")
    from comfy.utils import common_upscale
except Exception:
    has_load_image = False

logger = main_logger


def upscale(image, width, height, upscale_method):
    # return F.interpolate(image, size=(height, width), mode=upscale_method)
    return common_upscale(image, width, height, upscale_method, crop="disabled")


def upscale_comfy(image, width, height, upscale_method, crop="disabled"):
    if image.dim == 3:
        # A mask
        if upscale_method == "lanczos":
            # Lanczos needs an RGB image
            return upscale(image.unsqueeze(1).repeat(1, 3, 1, 1), width, height, upscale_method,
                           crop=crop).movedim(1, -1)[:, :, :, 0]
        else:
            return upscale(image.unsqueeze(1), width, height, upscale_method, crop=crop).squeeze(1)
    return common_upscale(image.movedim(-1, 1), width, height, upscale_method, crop=crop).movedim(1, -1)


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


def get_image_preview_info(file_name, where="input"):
    # This information is for the preview, as we are an output node and we return images
    # they will be displayed in our node. Quite simple.
    if os.path.isabs(file_name):
        base = get_input_directory() if where == "input" else get_output_directory()
        ff_name = os.path.relpath(file_name, base)
        fname = os.path.basename(ff_name)
        dname = os.path.dirname(ff_name)
    else:
        fname = file_name
        dname = ""
    return {"filename": fname, "subfolder": dname, "type": where}


def load_one_image(file_name, disp_name, embed_transparency):
    if os.path.isabs(file_name) and not os.path.exists(file_name):
        raise ValueError(f"File '{file_name}' not found")

    try:
        if has_load_image:
            # --- REUSE ComfyUI's LoadImage LOGIC ---
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
        return result

    except Exception as e:
        logger.error(f"Failed to load image '{disp_name}': {e}", exc_info=True)
        # Re-raise to make the error visible in ComfyUI
        raise IOError(f"Could not load the image file '{disp_name}'. "
                      "It may be corrupt or in an unsupported format.") from e


def load_image_wrapper(file_name, embed_transparency, disp_name=None, show_preview=True):
    disp_name = disp_name or file_name
    result = load_one_image(file_name, disp_name, embed_transparency)
    if not show_preview:
        return result
    return {"ui": {"images": [get_image_preview_info(file_name)]}, "result": result}


def load_images_wrapper(file_names, embed_transparency, disp_names=None, show_preview=True, batch_size=1):
    # We work with lists
    if isinstance(file_names, str):
        file_names = [file_names]
    disp_names = disp_names or file_names
    if isinstance(disp_names, str):
        disp_names = [disp_names]

    imgs = []
    masks = []
    all_preview_imgs = []
    total = len(file_names)
    for i in range(0, total, batch_size):
        if batch_size > 1:
            # Add upto batch_size images in a batch
            imgs_batch = []
            masks_batch = []
            max_w = max_h = 0
            max_mw = max_mh = 0
            for j in range(batch_size):
                index = i+j
                if index >= total:
                    continue
                file_name = file_names[i+j]
                disp_name = disp_names[i+j]

                img, mask = load_one_image(file_name, disp_name, embed_transparency)
                max_w = max(max_w, img.shape[2])
                max_h = max(max_h, img.shape[1])
                imgs_batch.append(img)
                max_mw = max(max_mw, mask.shape[2])
                max_mh = max(max_mh, mask.shape[1])
                masks_batch.append(mask)
                if show_preview:
                    all_preview_imgs.append(get_image_preview_info(file_name))
            for j in range(len(imgs_batch)):
                img = imgs_batch[j]
                H, W = img.shape[1:3]
                if H != max_h or W != max_w:
                    logger.debug(f"Upscaling image to fit batch: {W}x{H} -> {max_w}x{max_h}")
                    imgs_batch[j] = upscale_comfy(img, max_w, max_h, "bicubic")
                mask = masks_batch[j]
                H, W = mask.shape[1:3]
                if H != max_mh or W != max_mw:
                    logger.debug(f"Upscaling mask to fit batch: {W}x{H} -> {max_mw}x{max_mh}")
                    masks_batch[j] = upscale_comfy(mask, max_mw, max_mh, "bicubic")
            imgs.append(torch.cat(imgs_batch))
            masks.append(torch.cat(masks_batch))
        else:
            # Add a single image
            file_name = file_names[i]
            img, mask = load_one_image(file_name, disp_names[i], embed_transparency)
            imgs.append(img)
            masks.append(mask)
            if show_preview:
                all_preview_imgs.append(get_image_preview_info(file_name))
    logger.debug(f"Loaded {len(imgs)} batches:")
    for n, i in enumerate(imgs):
        logger.debug(f"{n}) {i.shape}")
    if not show_preview:
        return (imgs, masks)
    return {"ui": {"images": all_preview_imgs}, "result": (imgs, masks)}


def save_image(images, filenames, prompt=None, extra_pnginfo=None, compress_level=4, show_preview=True):
    if isinstance(filenames, str):
        if not filenames:
            raise ValueError("You must provide a file name")
        filenames = [filenames]

    # Make a list with all the images, unroll the batches
    imgs = []
    for img in images:
        imgs.extend([i for i in img])

    if len(filenames) != len(imgs):
        raise ValueError(f"{len(imgs)} images provided but only {len(filenames)} file names")

    all_preview_imgs = []
    for index, (image, filename) in enumerate(zip(imgs, filenames)):
        i = 255. * image.cpu().numpy()
        img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
        metadata = PngInfo()
        if prompt is not None:
            metadata.add_text("prompt", json.dumps(prompt))
        if extra_pnginfo is not None:
            for x in extra_pnginfo:
                metadata.add_text(x, json.dumps(extra_pnginfo[x]))

        logger.debug(f"Saving {image.shape[1]}x{image.shape[0]} image to {filename}")
        img.save(filename, pnginfo=metadata, compress_level=compress_level)

        if show_preview:
            all_preview_imgs.append(get_image_preview_info(filename, where="output"))

    if not show_preview:
        return ()
    return {"ui": {"images": all_preview_imgs}, "result": ()}
