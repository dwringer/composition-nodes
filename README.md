![composition pack main image](https://raw.githubusercontent.com/dwringer/composition-nodes/main/composition_pack.jpg)

### Installation:

To install these nodes, simply place the included `.py` files in your `invokeai/.venv/Lib/site-packages/invokeai/app/invocations/` or `invokeai/.venv/Lib/Python3.10/site-packages/invokeai/app/invocations/` folder, depending on which one exists on your system. You may also have a slightly different Python version listed. Navigate to your invokeai folder, open up .venv/Lib and you can locate the appropriate folder from there.

### Adjust Image Hue Plus

Rotate the hue of an image in one of several different color spaces.

### Blend Latents/Noise (Masked)

Use a mask to blend part of one latents/noise tensor into another. Can be used to "renoise" sections during a multi-stage [masked] denoising process.

### Enhance Image

Boost or reduce color saturation, contrast, brightness, sharpness, or invert colors of any image at any stage with this simple wrapper for pillow [PIL]'s ImageEnhance module.

Color inversion is toggled with a simple switch, while each of the four enhancer modes are activated by entering a value other than 1 in each corresponding input field. Values less than 1 will reduce the corresponding property, while values greater than 1 will enhance it.

**Example Usage:**
![enhance image usage graph](https://raw.githubusercontent.com/dwringer/composition-nodes/main/image_enhance_usage.jpg)

### Equivalent Achromatic Lightness

Calculates image lightness accounting for Helmholtz-Kohlrausch effect based on a method described by High, Green, and Nussbaum (2023) [https://doi.org/10.1002/col.22839].

### Text to Mask (Clipseg)

Input a prompt and an image to generate a mask representing areas of the image matched by the prompt.

### Text to Mask Advanced (Clipseg)

Output up to four prompt masks combined with logical "and", logical "or", or as separate channels of an RGBA image.

### Image Layer Blend

Perform a layered blend of two images using alpha compositing. Opacity of top layer is selectable, and a mask image may also be used. There are currently 23 blend modes supported and 8 color space modes. Four of the blend modes - Hue, Saturation, Color, and Luminosity - are restricted to only 6 of the color space modes: RGB and Linear RGB will convert to HSL for those blend modes. Several of the other blend modes only operate on the lightness channel of non-RGB color space modes.

Blend modes available: 
Normal, Lighten Only, Darken Only, Lighten Only (EAL), Darken Only (EAL), Hue, Saturation, Color, Luminosity, Linear Dodge (Add), Subtract, Multiply, Divide, Screen, Overlay, Linear Burn, Difference, Hard Light, Soft Light, Vivid Light, Linear Light, Color Burn, Color Dodge

Color space modes available:
RGB, Linear RGB, HSL, HSV, Okhsl, Okhsv, Oklch (Oklab), LCh (CIELab)

### Image Compositor

Take a subject from an image with a flat backdrop and layer it on another image using a chroma key to specify a color value/threshold to remove backdrop pixels, or leave the color blank and a "flood select" will be used from the image corners.

The subject image may be scaled using the fill X and fill Y options (enable both to stretch-fit).  Final subject position may also be adjusted with X offset and Y offset. If used, chroma key may be specified either as an (R, G, B) tuple, or a CSS-3 color string.

### Image Dilate or Erode

Dilate or expand a mask (or any image!). This is equivalent to an expand/contract operation.

### Image Value Thresholds

Clip an image to pure black/white beyond specified thresholds.

### Offset Latents

This takes a Latents input as well as two numbers (between 0 and 1), which are used to offset the latents in the vertical and/or horizontal directions. 0.5/0.5 would offset the image 50% in both directions such that the corners will wrap around and become the center of the image.

### Offset Image

This works like Offset Latents, but in image space, with the additional capability of taking exact pixel offsets instead of just percentages (toggled with a switch/boolean input).

### Rotate Image

Rotate an image in degrees about its center, clockwise (positive entries) or counterclockwise (negative entries). Optionally expand the image boundary to fit the rotated image.

### Shadows/Highlights/Midtones

Extract three masks (with adjustable hard or soft thresholds) representing shadows, midtones, and highlights regions of an image.

### Text Mask (simple 2D)

Create a white on black (or black on white) text image for use with controlnets or further processing in other nodes. Specify any TTF/OTF font file available to Invoke and control parameters to resize, rotate, and reposition the text.

Currently this only generates one line of text, but it can be layered with other images using the Image Compositor node or any other such tool.

### Example usage:

![composition nodes usage graph](https://raw.githubusercontent.com/dwringer/composition-nodes/main/composition_nodes_usage.jpg)
