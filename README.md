![composition nodes usage graph](https://raw.githubusercontent.com/dwringer/composition-nodes/main/composition_nodes_usage.jpg)

### Installation:

To install these nodes, simply place the included `.py` files in your `invokeai/.venv/Lib/site-packages/invokeai/app/invocations/` or `invokeai/.venv/Lib/Python3.10/site-packages/invokeai/app/invocations/` folder, depending on which one exists on your system. You may also have a slightly different Python version listed. Navigate to your invokeai folder, open up .venv/Lib and you can locate the appropriate folder from there.

### Image Compositor

Take a subject from an image with a flat backdrop and layer it on another image using a chroma key to specify a color value/threshold to remove backdrop pixels, or leave the color blank and a "flood select" will be used from the image corners.

The subject image may be scaled using the fill X and fill Y options (enable both to stretch-fit).  Final subject position may also be adjusted with X offset and Y offset. If used, chroma key may be specified either as an (R, G, B) tuple, or a CSS-3 color string.

### Text Mask (simple 2D)

Create a white on black (or black on white) text image for use with controlnets or further processing in other nodes. Specify any TTF/OTF font file available to Invoke and control parameters to resize, rotate, and reposition the text.

Currently this only generates one line of text, but it can be layered with other images using the Image Compositor node or any other such tool.

### Offset Latents

This takes a Latents input as well as two numbers (between 0 and 1), which are used to offset the latents in the vertical and/or horizontal directions. 0.5/0.5 would offset the image 50% in both directions such that the corners will wrap around and become the center of the image.

### Offset Image

This works like Offset Latents, but in image space, with the additional capability of taking exact pixel offsets instead of just percentages (toggled with a switch/boolean input).
