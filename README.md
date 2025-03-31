### Note:

Currently, the version of the Transformers library [4.46.3] that's pinned for the InvokeAI package has a regression which results in the Clipseg nodes failing to work properly, giving the error ("ValueError: Input image size (352x352) doesn't match model (224x224)."). This is fixed at least as early as Transformers 4.48.3, which can be installed by activating your InvokeAI .venv and typing `uv pip install transformers==4.48.3`. This will fix the Clipseg nodes, although it's always possible there might be unintended consequences from upgrading.


### Installation:

To install these nodes, simply place the folder containing this repository's code (or just clone the repository yourself) into your `invokeai/nodes` folder.

#### Text to Mask (Clipseg)

Input a prompt and an image to generate a mask representing areas of the image matched by the prompt.

#### Text to Mask Advanced (Clipseg)

Output up to four prompt masks combined with logical "and", logical "or", or as separate channels of an RGBA image.

#### Image Search to Mask (Clipseg)

Input a base image and a prompt image to generate a mask representing areas of the base image matched by the prompt image contents.

#### Clipseg Mask Hierarchy

This node takes up to seven pairs of prompts/threshold values, then descends through them hierarchically creating mutually exclusive masks out of whatever it can match from the input image. This means whatever is matched in prompt 1 will be subtracted from the match area for prompt 2; both areas will be omitted from the match area of prompt 3; etc. The idea is that by starting with foreground objects and working your way back through a scene, you can create a more-or-less complete segmentation map for the image whose constituent segments can be passed off to different masks for regional conditioning or other processing.

#### Offset Latents

This takes a Latents input as well as two numbers (between 0 and 1), which are used to offset the latents in the vertical and/or horizontal directions. 0.5/0.5 would offset the image 50% in both directions such that the corners will wrap around and become the center of the image.

#### Offset Image

This works like Offset Latents, but in image space, with the additional capability of taking exact pixel offsets instead of just percentages (toggled with a switch/boolean input).

#### Rotate/Flip Image

Rotate an image in degrees about its center, clockwise (positive entries) or counterclockwise (negative entries). Optionally expand the image boundary to fit the rotated image, or flip it horizontally or vertically.

#### Shadows/Highlights/Midtones

Extract three masks (with adjustable hard or soft thresholds) representing shadows, midtones, and highlights regions of an image.

#### Text Mask (simple 2D)

Create a white on black (or black on white) text image for use with controlnets or further processing in other nodes. Specify any TTF/OTF font file available to Invoke and control parameters to resize, rotate, and reposition the text.

Currently this only generates one line of text, but it can be layered with other images using the Image Compositor node or any other such tool.

#### 2D Noise Image

Creates an image of 2D Noise approximating the desired characteristics, using various combinations of gaussian blur and arithmetic operations to perform low pass and high pass filtering of 2-dimensional spatial frequencies of each channel to create Red, Blue, or Green "colored noise".

#### Noise (Spectral Characteristics)

This operates like 2D Noise Image but outputs latent tensors, 4-channel or 16-channel.

#### Flatten Histogram (Grayscale)

Scales the values of an L-mode image by scaling them to the full range 0..255 in equal proportions

#### Add Noise (Flux)

Calculates the correct initial timestep noising amount and applies it to the given latent tensor using simple addition according to the specified ratio.

#### Latent Quantize (Kohonen map)

Use a self-organizing map to quantize the values of a latent tensor. This is highly experimental and not really suitable for most use cases. It's very easy to use settings that will appear to hang, or tie up the PC for a very long time, so use of this node is somewhat discouraged.

#### Image Quantize (Kohonen map)

Use a Kohonen self-organizing map to quantize the pixel values of an image. It's possible to pass in an existing map, which will be used instead of training a new one. It's also possible to pass in a "swap map", which will be used in place of the standard map's assigned pixel values in quantizing the target image - these values can be correlated either one by one by a linear assignment minimizing the distances\* between each of them, or by swapping their coordinates on the maps themselves, which can be orientedfirst such that their corner distances\* are minimized achieving a closest-fit while attempting to preserve mappings of adjacent colors.

\*Here, "distances" refers to the euclidean distances between (L, a, b) tuples in Oklab color space.

### Nodes since moved to Invoke core:

The following nodes are no longer maintained here because they were integrated with InvokeAI core. If you have a version prior to 5.4.2 and want to install them anyway, you can revert to the pre-invoke-5_4 tag of this repo.

#### Adjust Image Hue Plus

Rotate the hue of an image in one of several different color spaces.

#### Blend Latents/Noise (Masked)

Use a mask to blend part of one latents/noise tensor into another. Can be used to "renoise" sections during a multi-stage [masked] denoising process.

#### Enhance Image

Boost or reduce color saturation, contrast, brightness, sharpness, or invert colors of any image at any stage with this simple wrapper for pillow [PIL]'s ImageEnhance module.

Color inversion is toggled with a simple switch, while each of the four enhancer modes are activated by entering a value other than 1 in each corresponding input field. Values less than 1 will reduce the corresponding property, while values greater than 1 will enhance it.

**Example Usage:**
![enhance image usage graph](https://raw.githubusercontent.com/dwringer/composition-nodes/main/image_enhance_usage.jpg)

#### Equivalent Achromatic Lightness

Calculates image lightness accounting for Helmholtz-Kohlrausch effect based on a method described by High, Green, and Nussbaum (2023) [https://doi.org/10.1002/col.22839].

#### Image Layer Blend

Perform a layered blend of two images using alpha compositing. Opacity of top layer is selectable, and a mask image may also be used. There are currently 23 blend modes supported and 8 color space modes. Four of the blend modes - Hue, Saturation, Color, and Luminosity - are restricted to only 6 of the color space modes: RGB and Linear RGB will convert to HSL for those blend modes. Several of the other blend modes only operate on the lightness channel of non-RGB color space modes.

Blend modes available: 
Normal, Lighten Only, Darken Only, Lighten Only (EAL), Darken Only (EAL), Hue, Saturation, Color, Luminosity, Linear Dodge (Add), Subtract, Multiply, Divide, Screen, Overlay, Linear Burn, Difference, Hard Light, Soft Light, Vivid Light, Linear Light, Color Burn, Color Dodge

Color space modes available:
RGB, Linear RGB, HSL, HSV, Okhsl, Okhsv, Oklch (Oklab), LCh (CIELab)

#### Image Compositor

Take a subject from an image with a flat backdrop and layer it on another image using a chroma key to specify a color value/threshold to remove backdrop pixels, or leave the color blank and a "flood select" will be used from the image corners.

The subject image may be scaled using the fill X and fill Y options (enable both to stretch-fit).  Final subject position may also be adjusted with X offset and Y offset. If used, chroma key may be specified either as an (R, G, B) tuple, or a CSS-3 color string.

#### Image Dilate or Erode

Dilate or expand a mask (or any image!). This is equivalent to an expand/contract operation.

#### Image Value Thresholds

Clip an image to pure black/white beyond specified thresholds.

#### Example usage:

![composition nodes usage graph](https://raw.githubusercontent.com/dwringer/composition-nodes/main/composition_nodes_usage.jpg)
