# ComfyUI Image Miscellaneous Nodes &#x0001f3a8;

This repository provides a set of custom nodes for ComfyUI focused on image manipulation.
Currently we just have a few nodes used by other nodes I maintain.


## &#x2699;&#xFE0F; Main features

&#x2705; No bizarre extra dependencies, we use the same modules as ComfyUI

&#x2705; Warnings and errors visible in the browser, configurable debug information in the console


## &#x0001F4DC; Table of Contents

- &#x0001F680; [Installation](#-installation)
- &#x0001F4E6; [Dependencies](#-dependencies)
- &#x0001F5BC;&#xFE0F; [Examples](#&#xFE0F;-examples)
- &#x2728; [Nodes](#-extra-nodes)
  - [Image Download and Load](#1-image-download-and-load)
  - [Face Composite](#2-face-composite)
  - [Face Composite (frame by frame)](#3-face-composite-frame-by-frame)
- &#x0001F4DD; [Usage Notes](#-usage-notes)
- &#x0001F4DC; [Project History](#-project-history)
- &#x2696;&#xFE0F; [License](#&#xFE0F;-license)
- &#x0001F64F; [Attributions](#-attributions)

## &#x2728; Nodes

### 1. Image Download and Load
   - **Display Name:** `Image Download and Load`
   - **Internal Name:** `SET_ImageDownload`
   - **Category:** `image/io`
   - **Description:** Downloads an image file from a URL into the `ComfyUI/input/` directory if it's not already there, and then loads it as an image and mask. This is perfect for creating self-contained, shareable workflows with example image.
   - **Inputs:**
     - `image_bypass` (IMAGE, Optional): If an image is provided here it will be used for the output. You can connect a `Load Image` node here, if the connected node is muted (bypassed) we download the file, otherwise we use the image from the `Load Image` node.
     - `mask_bypass` (MASK, Optional): This input complements `image_bypass`.
     - `base_url` (STRING): The URL of the directory containing the image file.
     - `filename` (STRING): The name of the image file to download (e.g., photo.jpg, art.png).
   - **Output:**
     - `image` (IMAGE): The loaded image.
     - `alpha_mask` (MASK): The alpha mask for the loaded image.
   - **Behavior Details:**
     - **Caching:** The node checks the `ComfyUI/input/` folder first. If the file with the specified `filename` already exists, the download is skipped.
     - **Bypass:** If only one of `image_bypass` and `mask_bypass` is connected the other will be assumed to be empty. You should connect both or avoid using the output corresponding to the unconnected input.

### 2. Face Composite
   - **Display Name:** `Face Composite`
   - **Internal Name:** `SET_CompositeFace`
   - **Category:** `image/manipulation`
   - **Description:** This node is designed for complex batch operations where a single reference image generates multiple animated or processed frames. It handles an **M-to-N** relationship.
   - **Purpose:** The primary use case is for animation workflows (e.g., using AnimateDiff or SVD) where you:
      1.  Extract a face from **one** source image.
      2.  Generate **N** animated frames of that face.
      3.  Paste each of the **N** animated frames back onto the original source image to create an animated sequence.
   - **Inputs:**
     - `animated` (`IMAGE`): A batch of `M*N` cropped/processed images (e.g., the animated faces).
     - `reference` (`IMAGE`): A batch of `M` original context images that the faces were extracted from.
     - `bboxes` (`BBOX`): A list of `M` bounding boxes. Each box in the list must correspond to a `reference` image, defining the location for the paste operation. Format: (X, Y, W, H)
   - **Output:**
     - `images` (`IMAGE`): The final output batch of `M*N` full-size images. Each image consists of a `reference` image with one of the `animated` frames pasted onto it.
   - **How it Works:**
     - For each of the `M` reference images and its corresponding bounding box, the node takes the next `N` frames from the `animated` batch. It then pastes each of these `N` frames onto a *copy* of the reference image, generating `N` final output frames. This process is repeated for all `M` reference images.
     - The animated frame is automatically resized to fit the bounding box dimensions before pasting. The pasting logic safely handles cases where the bbox is partially outside the image boundaries.

### 3. Face Composite (frame by frame)
   - **Display Name:** `"Face Composite (frame by frame)`
   - **Internal Name:** `SET_CompositeFaceFrameByFrame`
   - **Category:** `image/manipulation`
   - **Description:** This node is a simplified variant for **1-to-1** compositing. It is perfect for video processing workflows where you need to paste a sequence of processed frames back into the original video sequence at a static location.
   - **Purpose:** Ideal for tasks like video face restoration, stylization, or simple lip-sync, where:
      1.  A video is loaded as a sequence of `N` frames.
      2.  A corresponding sequence of `N` processed/animated frames is generated.
      3.  Each processed frame needs to be pasted back into its corresponding original frame at the same location.
   - **Inputs:**
     - `animated` (`IMAGE`): A batch of `N` processed frames.
     - `reference` (`IMAGE`): A batch of `N` original frames. **Must have the same batch size as `animated`**.
     - `bboxes` (`BBOX`): A list of bounding boxes. **Only the first bbox in the list is used** as the static paste location for all frames.
   - **Output:**
     - `images` (`IMAGE`): The final batch of `N` composited frames.
   - **How it Works:** The node iterates from frame 0 to N-1. In each step, it takes the i-th `animated` frame and the i-th `reference` frame. It then pastes the animated frame onto the reference frame using the coordinates from the single, static bounding box.


## &#x0001F680; Installation

You can install the nodes from the ComfyUI nodes manager, the name is *Image Misc*, or just do it manually:

1.  Clone this repository into your `ComfyUI/custom_nodes/` directory:
    ```bash
    cd ComfyUI/custom_nodes/
    git clone https://github.com/set-soft/ComfyUI-ImageMisc ComfyUI-ImageMisc
    ```
2.  Install dependencies: `pip install -r ComfyUI/custom_nodes/ComfyUI-ImageMisc/requirements.txt`
3.  Restart ComfyUI.

The nodes should then appear under the "image/io" category in the "Add Node" menu.


## &#x0001F4E6; Dependencies

- SeCoNoHe (seconohe): This is just some functionality I wrote shared by my nodes, only depends on ComfyUI.
- PyTorch: Installed by ComfyUI
- NumPy: Installed by ComfyUI
- Pillow: Installed by ComfyUI
- Requests (optional): Usually an indirect ComfyUI dependency. If installed it will be used for downloads, it should be more robust than then built-in `urllib`, used as fallback.
- Colorama (optional): Might help to get colored log messages on some terminals. We use ANSI escape sequences when it isn't installed.


## &#x0001F5BC;&#xFE0F; Examples

Once installed the examples are available in the ComfyUI workflow templates, in the *Image Misc* section (or ComfyUI-ImageMisc).

- [image_download.json](example_workflows/image_download.json): Shows how to use the image downloader node.


## &#x0001F4DD; Usage Notes

- **Logging:** &#x0001F50A; The nodes use Python's `logging` module. Debug messages can be helpful for understanding the transformations being applied.
  You can control log verbosity through ComfyUI's startup arguments (e.g., `--preview-method auto --verbose DEBUG` for more detailed ComfyUI logs
  which might also affect custom node loggers if they are configured to inherit levels). The logger name used is "ImageMisc".
  You can force debugging level for these nodes defining the `IMAGEMISC_NODES_DEBUG` environment variable to `1`.


## &#x0001F4DC; Project History

- 1.0.0 2025-07-21: Initial release
  - Just the download image.


## &#x2696;&#xFE0F; License

[GPL-3.0](LICENSE)

## &#x0001F64F; Attributions

- Good part of the initial code and this README was generated using Gemini 2.5 Pro.
