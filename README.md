# ComfyUI Image Miscellaneous Nodes &#x0001f3a8;

This repository provides a set of custom nodes for ComfyUI focused on image manipulation.
Currently we just have an image downloader node, but I didn't wanted to put it with
my audio nodes.


## &#x2699;&#xFE0F; Main features

&#x2705; No extra dependencies, we use the same modules as ComfyUI

&#x2705; Warnings and errors visible in the browser, configurable debug information in the console


## &#x0001F4DC; Table of Contents

- &#x0001F680; [Installation](#-installation)
- &#x0001F4E6; [Dependencies](#-dependencies)
- &#x0001F5BC;&#xFE0F; [Examples](#&#xFE0F;-examples)
- &#x2728; [Nodes](#-extra-nodes)
  - [1. Image Download and Load](#1-image-download-and-load)
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


## &#x0001F680; Installation

You can install the nodes from the ComfyUI nodes manager, the name is *Image Misc*, or just do it manually:

1.  Clone this repository into your `ComfyUI/custom_nodes/` directory:
    ```bash
    cd ComfyUI/custom_nodes/
    git clone https://github.com/set-soft/ComfyUI-ImageMisc ComfyUI-ImageMisc
    ```
2.  Install SeCoNoHe: `pip install seconohe`
3.  Restart ComfyUI.

The nodes should then appear under the "image/io" category in the "Add Node" menu.


## &#x0001F4E6; Dependencies

- SeCoNoHe (seconohe): This is just some functionality I wrote shared by my nodes, only depends on ComfyUI.
- PyTorch: Installed by ComfyUI
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
