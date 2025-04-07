# composition-nodes

**Repository Name:** Composition Nodes Pack for InvokeAI

**Author:** dwringer

**License:** MIT

**Requirements:**
- invokeai>=4

## Introduction
### Note:

If you're attempting to use any of the Clipseg nodes and you
encounter the error, `ValueError: Input image size (352x352) doesn't
match model (224x224).`, it is because there was a temporary
regression in the Transformers library that broke Clipseg pipelines
for a few versions, which got installed with certain
versions/platforms of InvokeAI. This issue was fixed at least as
early as Transformers 4.48.3, which can be installed by activating
your InvokeAI .venv and typing `uv pip install
transformers==4.48.3`.

### Installation:

To install these nodes, simply place the folder containing this
repository's code (or just clone the repository yourself) into your
`invokeai/nodes` folder.

Generally, the two methods of installation are:

- Open a terminal with git access (`git bash` on Windows) in
your InvokeAI home directory, and `cd` to the `nodes`
subdirectory. If you installed to `C:\Users\<user
name>\invokeai\` then you will want your terminal to be open to
`C:\Users\<user name>\invokeai\nodes\`.  Then simply type:
```
git clone https://github.com/dwringer/composition-nodes.git
```

- Or, download the source code of this repository as a .zip file by
clicking the green `<> Code` button above and selecting `Download
ZIP`, then extract the folder within and place it as a subfolder
inside the `nodes` folder of your InvokeAI home directory (e.g.,
`C:\Users\<user name>\invokeai\nodes\composition-nodes-master\`)

## Overview
### Nodes
- [2D Noise Image](#2d-noise-image) - Creates an image of 2D Noise approximating the desired characteristics.
- [Add Noise (Flux)](#add-noise-flux) - Add noise to a flux latents tensor using the appropriate ratio given the denoising schedule timestep.
- [Clipseg Mask Hierarchy](#clipseg-mask-hierarchy) - Creates a segmentation hierarchy of mutually exclusive masks from clipseg text prompts.
- [CMYK Color Separation](#cmyk-color-separation) - Get color images from a base color and two others that subtractively mix to obtain it
- [CMYK Merge](#cmyk-merge) - Merge subtractive color channels (CMYK+alpha)
- [CMYK Split](#cmyk-split) - Split an image into subtractive color channels (CMYK+alpha)
- [Flatten Histogram (Grayscale)](#flatten-histogram-grayscale) - Scales the values of an L-mode image by scaling them to the full range 0..255 in equal proportions
- [Freq. Blend Match & Phase Blend](#freq-blend-match--phase-blend) - Generates latent noise with the frequency spectrum of target latents, masked by a provided mask image.
- [Freq. Match & Phase Blend Latents](#freq-match--phase-blend-latents) - Generates latent noise with the frequency spectrum of target latents, masked by a provided mask image.
- [Frequency Spectrum Match Latents](#frequency-spectrum-match-latents) - Generates latent noise with the frequency spectrum of target latents, masked by a provided mask image.
- [Image Quantize (Kohonen map)](#image-quantize-kohonen-map) - Use a Kohonen self-organizing map to quantize the pixel values of an image.
- [Image Search to Mask (Clipseg)](#image-search-to-mask-clipseg) - Uses the Clipseg model to generate an image mask from an image prompt.
- [Latent Quantize (Kohonen map)](#latent-quantize-kohonen-map) - Use a self-organizing map to quantize the values of a latent tensor.
- [Noise (Spectral characteristics)](#noise-spectral-characteristics) - Creates a latents tensor of 2D noise channels approximating the desired characteristics.
- [Offset Image](#offset-image) - Offsets an image by a given percentage (or pixel amount).
- [Offset Latents](#offset-latents) - Offsets a latents tensor by a given percentage of height/width.
- [RGB Merge](#rgb-merge) - Merge RGB color channels and alpha
- [RGB Split](#rgb-split) - Split an image into RGB color channels and alpha
- [Rotate/Flip Image](#rotateflip-image) - Rotates an image by a given angle (in degrees clockwise).
- [Shadows/Highlights/Midtones](#shadowshighlightsmidtones) - Extract a Shadows/Highlights/Midtones mask from an image.
- [Text Mask](#text-mask) - Creates a 2D rendering of a text mask from a given font.
- [Text to Mask (Clipseg)](#text-to-mask-clipseg) - Uses the Clipseg model to generate an image mask from a text prompt.
- [Text to Mask Advanced (Clipseg)](#text-to-mask-advanced-clipseg) - Uses the Clipseg model to generate an image mask from a text prompt.

<details>
<summary>

### Functions

</summary>

- `load_profiles` - Load available ICC profile filenames from COLOR_PROFILES_DIR into a dict
- `tensor_from_pil_image` - 
- `flatten_histogram` - 
- `white_noise_array` - 
- `white_noise_image` - 
- `red_noise_array` - 
- `red_noise_image` - 
- `blue_noise_array` - 
- `blue_noise_image` - 
- `green_noise_array` - 
- `green_noise_image` - 
- `tensor_from_pil_image` - 
- `calculate_corner_distances` - Calculate Euclidean distances between corresponding corners' [Ok]LAB color coordinates.
- `get_image_corners` - Extract corner pixel RGB values from an image.
- `find_best_transformation` - Find the best image rotation/flip transformation to minimize corner distances.
- `apply_transformation` - Apply a preset transformation to an image.
</details>

<details>
<summary>

### Output Definitions

</summary>

- `ClipsegMaskHierarchyOutput` - Output definition with 8 fields
- `CMYKSplitOutput` - Output definition with 7 fields
- `CMYKSeparationOutput` - Output definition with 11 fields
- `ImageSOMOutput` - Output definition with 6 fields
- `RGBSplitOutput` - Output definition with 6 fields
- `ShadowsHighlightsMidtonesMasksOutput` - Output definition with 5 fields
</details>

## Nodes
### 2D Noise Image
**ID:** `noiseimg_2d`

**Category:** image

**Tags:** image, noise

**Version:** 1.2.0

**Description:** Creates an image of 2D Noise approximating the desired characteristics.

Creates an image of 2D Noise approximating the desired characteristics, using various combinations of gaussian blur and arithmetic operations to perform low pass and high pass filtering of 2-dimensional spatial frequencies of each channel to create Red, Blue, or Green "colored noise".

<details>
<summary>

#### Inputs

</summary>

| Name | Type | Description | Default |
| ---- | ---- | ----------- | ------- |
| `noise_type` | `Literal[(White, Red, Blue, Green)]` | Desired noise spectral characteristics | White |
| `width` | `int` | Desired image width | 512 |
| `height` | `int` | Desired image height | 512 |
| `seed` | `int` | Seed for noise generation | 0 |
| `iterations` | `int` | Noise approx. iterations | 15 |
| `blur_threshold` | `float` | Threshold used in computing noise (lower is better/slower) | 0.2 |
| `sigma_red` | `float` | Sigma for strong gaussian blur LPF for red/green | 3.0 |
| `sigma_blue` | `float` | Sigma for weak gaussian blur HPF for blue/green | 1.0 |


</details>

<details>
<summary>

#### Output

</summary>

**Type:** `ImageOutput.build(...)`



</details>

---
### Add Noise (Flux)
**ID:** `noise_add_flux`

**Category:** noise

**Tags:** latents, blend, noise

**Version:** 0.0.1

**Description:** Add noise to a flux latents tensor using the appropriate ratio given the denoising schedule timestep.

Calculates the correct initial timestep noising amount and applies it to the given latent tensor using simple addition according to the specified ratio.

<details>
<summary>

#### Inputs

</summary>

| Name | Type | Description | Default |
| ---- | ---- | ----------- | ------- |
| `latents_in` | `LatentsField` |  | None |
| `noise_in` | `LatentsField` | The noise to be added | None |
| `num_steps` | `int` | Number of diffusion steps | None |
| `denoising_start` | `float` | Starting point for denoising (0.0 to 1.0) | None |
| `is_schnell` | `bool` | Boolean flag indicating if this is a FLUX Schnell model | None |


</details>

<details>
<summary>

#### Output

</summary>

**Type:** `LatentsOutput.build(...)`



</details>

---
### Clipseg Mask Hierarchy
**ID:** `clipseg_mask_hierarchy`

**Category:** image

**Tags:** image, mask, clip, clipseg, txt2mask, hierarchy

**Version:** 1.2.2

**Description:** Creates a segmentation hierarchy of mutually exclusive masks from clipseg text prompts.

This node takes up to seven pairs of prompts/threshold values, then descends through them hierarchically creating mutually exclusive masks out of whatever it can match from the input image. This means whatever is matched in prompt 1 will be subtracted from the match area for prompt 2; both areas will be omitted from the match area of prompt 3; etc. The idea is that by starting with foreground objects and working your way back through a scene, you can create a more-or-less complete segmentation map for the image whose constituent segments can be passed off to different masks for regional conditioning or other processing.

<details>
<summary>

#### Inputs

</summary>

| Name | Type | Description | Default |
| ---- | ---- | ----------- | ------- |
| `image` | `ImageField` | The image from which to create masks | None |
| `invert_output` | `bool` | Off: white on black / On: black on white | True |
| `smoothing` | `float` | Radius of blur to apply before thresholding | 4.0 |
| `prompt_1` | `str` | Text to mask prompt with highest segmentation priority | None |
| `threshold_1` | `float` | Detection confidence threshold for prompt 1 | 0.4 |
| `prompt_2` | `str` | Text to mask prompt, behind prompt 1 | None |
| `threshold_2` | `float` | Detection confidence threshold for prompt 2 | 0.4 |
| `prompt_3` | `str` | Text to mask prompt, behind prompts 1 & 2 | None |
| `threshold_3` | `float` | Detection confidence threshold for prompt 3 | 0.4 |
| `prompt_4` | `str` | Text to mask prompt, behind prompts 1, 2, & 3 | None |
| `threshold_4` | `float` | Detection confidence threshold for prompt 4 | 0.4 |
| `prompt_5` | `str` | Text to mask prompt, behind prompts 1 thru 4 | None |
| `threshold_5` | `float` | Detection confidence threshold for prompt 5 | 0.4 |
| `prompt_6` | `str` | Text to mask prompt, behind prompts 1 thru 5 | None |
| `threshold_6` | `float` | Detection confidence threshold for prompt 6 | 0.4 |
| `prompt_7` | `str` | Text to mask prompt, lowest priority behind all others | None |
| `threshold_7` | `float` | Detection confidence threshold for prompt 7 | 0.4 |


</details>

<details>
<summary>

#### Output

</summary>

**Type:** `ClipsegMaskHierarchyOutput`

| Name | Type | Description |
| ---- | ---- | ----------- |
| `mask_1` | `ImageField` | Mask corresponding to prompt 1 (full coverage) |
| `mask_2` | `ImageField` | Mask corresponding to prompt 2 (minus mask 1) |
| `mask_3` | `ImageField` | Mask corresponding to prompt 3 (minus masks 1 & 2) |
| `mask_4` | `ImageField` | Mask corresponding to prompt 4 (minus masks 1, 2, & 3) |
| `mask_5` | `ImageField` | Mask corresponding to prompt 5 (minus masks 1 thru 4) |
| `mask_6` | `ImageField` | Mask corresponding to prompt 6 (minus masks 1 thru 5) |
| `mask_7` | `ImageField` | Mask corresponding to prompt 7 (minus masks 1 thru 6) |
| `ground_mask` | `ImageField` | Mask coresponding to remaining unmatched image areas. |


</details>

---
### CMYK Color Separation
**ID:** `cmyk_separation`

**Category:** image

**Tags:** image, cmyk, separation, color

**Version:** 1.2.0

**Description:** Get color images from a base color and two others that subtractively mix to obtain it

<details>
<summary>

#### Inputs

</summary>

| Name | Type | Description | Default |
| ---- | ---- | ----------- | ------- |
| `width` | `int` | Desired image width | 512 |
| `height` | `int` | Desired image height | 512 |
| `c_value` | `float` | Desired final cyan value | 0 |
| `m_value` | `float` | Desired final magenta value | 25 |
| `y_value` | `float` | Desired final yellow value | 28 |
| `k_value` | `float` | Desired final black value | 76 |
| `c_split` | `float` | Desired cyan split point % [0..1.0] | 0.5 |
| `m_split` | `float` | Desired magenta split point % [0..1.0] | 1.0 |
| `y_split` | `float` | Desired yellow split point % [0..1.0] | 0.0 |
| `k_split` | `float` | Desired black split point % [0..1.0] | 0.5 |
| `profile` | `Literal[tuple(Any)]` | CMYK Color Profile | None |


</details>

<details>
<summary>

#### Output

</summary>

**Type:** `CMYKSeparationOutput`

| Name | Type | Description |
| ---- | ---- | ----------- |
| `color_image` | `ImageField` | Blank image of the specified color |
| `width` | `int` | The width of the image in pixels |
| `height` | `int` | The height of the image in pixels |
| `part_a` | `ImageField` | Blank image of the first separated color |
| `rgb_red_a` | `int` | R value of color part A |
| `rgb_green_a` | `int` | G value of color part A |
| `rgb_blue_a` | `int` | B value of color part A |
| `part_b` | `ImageField` | Blank image of the second separated color |
| `rgb_red_b` | `int` | R value of color part B |
| `rgb_green_b` | `int` | G value of color part B |
| `rgb_blue_b` | `int` | B value of color part B |


</details>

---
### CMYK Merge
**ID:** `cmyk_merge`

**Category:** image

**Tags:** cmyk, image, color

**Version:** 1.2.0

**Description:** Merge subtractive color channels (CMYK+alpha)

<details>
<summary>

#### Inputs

</summary>

| Name | Type | Description | Default |
| ---- | ---- | ----------- | ------- |
| `c_channel` | `Optional[ImageField]` | The c channel | None |
| `m_channel` | `Optional[ImageField]` | The m channel | None |
| `y_channel` | `Optional[ImageField]` | The y channel | None |
| `k_channel` | `Optional[ImageField]` | The k channel | None |
| `alpha_channel` | `Optional[ImageField]` | The alpha channel | None |
| `profile` | `Literal[tuple(Any)]` | CMYK Color Profile | None |


</details>

<details>
<summary>

#### Output

</summary>

**Type:** `ImageOutput.build(...)`



</details>

---
### CMYK Split
**ID:** `cmyk_split`

**Category:** image

**Tags:** cmyk, image, color

**Version:** 1.2.0

**Description:** Split an image into subtractive color channels (CMYK+alpha)

<details>
<summary>

#### Inputs

</summary>

| Name | Type | Description | Default |
| ---- | ---- | ----------- | ------- |
| `image` | `ImageField` | The image to split into additive channels | None |
| `profile` | `Literal[tuple(Any)]` | CMYK Color Profile | None |


</details>

<details>
<summary>

#### Output

</summary>

**Type:** `CMYKSplitOutput`

| Name | Type | Description |
| ---- | ---- | ----------- |
| `c_channel` | `ImageField` | Grayscale image of the cyan channel |
| `m_channel` | `ImageField` | Grayscale image of the magenta channel |
| `y_channel` | `ImageField` | Grayscale image of the yellow channel |
| `k_channel` | `ImageField` | Grayscale image of the k channel |
| `alpha_channel` | `ImageField` | Grayscale image of the alpha channel |
| `width` | `int` | The width of the image in pixels |
| `height` | `int` | The height of the image in pixels |


</details>

---
### Flatten Histogram (Grayscale)
**ID:** `flatten_histogram_mono`

**Category:** image

**Tags:** noise, image

**Version:** 1.2.0

**Description:** Scales the values of an L-mode image by scaling them to the full range 0..255 in equal proportions

<details>
<summary>

#### Inputs

</summary>

| Name | Type | Description | Default |
| ---- | ---- | ----------- | ------- |
| `image` | `ImageField` | Single-channel image for which to flatten the histogram | None |


</details>

<details>
<summary>

#### Output

</summary>

**Type:** `ImageOutput.build(...)`



</details>

---
### Freq. Blend Match & Phase Blend
**ID:** `frequency_blend_match_phase_blend_latents`

**Category:** latents

**Tags:** latents, noise, frequency, mask, blend, phase

**Version:** 1.0.0

**Description:** Generates latent noise with the frequency spectrum of target latents, masked by a provided mask image.

Takes both target latents and white noise latents as inputs.

<details>
<summary>

#### Inputs

</summary>

| Name | Type | Description | Default |
| ---- | ---- | ----------- | ------- |
| `target_latents_a` | `LatentsField` | Target latents to match frequency spectrum (A) | None |
| `target_latents_b` | `LatentsField` | Target latents to match frequency spectrum (B) | None |
| `frequency_blend_alpha` | `float` | Blend ratio for the frequency spectra | 0.0 |
| `phase_latents_a` | `LatentsField` | White noise latents for phase information (A) | None |
| `phase_latents_b` | `LatentsField` | White noise latents for phase information (B) | None |
| `phase_blend_alpha` | `float` | Blend ratio for the phases | 0.0 |
| `mask` | `Optional[ImageField]` | Mask for blending (optional) | None |
| `blur_sigma` | `float` | Amount of Gaussian blur to apply to the mask | 0 |


</details>

<details>
<summary>

#### Output

</summary>

**Type:** `LatentsOutput.build(...)`



</details>

---
### Freq. Match & Phase Blend Latents
**ID:** `frequency_match_phase_blend_latents`

**Category:** latents

**Tags:** latents, noise, frequency, mask, blend, phase

**Version:** 1.0.0

**Description:** Generates latent noise with the frequency spectrum of target latents, masked by a provided mask image.

Takes both target latents and white noise latents as inputs.

<details>
<summary>

#### Inputs

</summary>

| Name | Type | Description | Default |
| ---- | ---- | ----------- | ------- |
| `target_latents` | `LatentsField` | Target latents to match frequency spectrum | None |
| `phase_latents_a` | `LatentsField` | White noise latents for phase information (A) | None |
| `phase_latents_b` | `LatentsField` | White noise latents for phase information (B) | None |
| `phase_blend_alpha` | `float` | Blend ratio for the phases | 0.0 |
| `mask` | `Optional[ImageField]` | Mask for blending (optional) | None |
| `blur_sigma` | `float` | Amount of Gaussian blur to apply to the mask | 0 |


</details>

<details>
<summary>

#### Output

</summary>

**Type:** `LatentsOutput.build(...)`



</details>

---
### Frequency Spectrum Match Latents
**ID:** `frequency_match_latents`

**Category:** latents

**Tags:** latents, noise, frequency, mask, blend

**Version:** 1.0.0

**Description:** Generates latent noise with the frequency spectrum of target latents, masked by a provided mask image.

Takes both target latents and white noise latents as inputs.

<details>
<summary>

#### Inputs

</summary>

| Name | Type | Description | Default |
| ---- | ---- | ----------- | ------- |
| `target_latents` | `LatentsField` | Target latents to match frequency spectrum | None |
| `white_noise_latents` | `LatentsField` | White noise latents for phase information | None |
| `mask` | `Optional[ImageField]` | Mask for blending (optional) | None |
| `blur_sigma` | `float` | Amount of Gaussian blur to apply to the mask | 0 |


</details>

<details>
<summary>

#### Output

</summary>

**Type:** `LatentsOutput.build(...)`



</details>

---
### Image Quantize (Kohonen map)
**ID:** `image_som`

**Category:** image

**Tags:** image, color, quantize, som, kohonen, palette

**Version:** 0.8.2

**Description:** Use a Kohonen self-organizing map to quantize the pixel values of an image.

It's possible to pass in an existing map, which will be used instead of training a new one. It's also possible to pass in a "swap map", which will be used in place of the standard map's assigned pixel values in quantizing the target image - these values can be correlated either one by one by a linear assignment minimizing the distances\* between each of them, or by swapping their coordinates on the maps themselves, which can be oriented first such that their corner distances\* are minimized achieving a closest-fit while attempting to preserve mappings of adjacent colors.

\*Here, "distances" refers to the euclidean distances between (L, a, b) tuples in Oklab color space.

<details>
<summary>

#### Inputs

</summary>

| Name | Type | Description | Default |
| ---- | ---- | ----------- | ------- |
| `image_in` | `ImageField` | The image to quantize | None |
| `map_in` | `Optional[ImageField]` | Use an existing SOM instead of training one (skips all training) | None |
| `swap_map` | `Optional[ImageField]` | Take another map and swap in its colors after obtaining best-match indices but prior to mapping | None |
| `swap_mode` | `Literal[tuple(SWAP_MODES)]` | How to employ the swap map - directly, reoriented or rearranged | None |
| `map_width` | `int` | Width (in cells) of the self-organizing map to train | 16 |
| `map_height` | `int` | Height (in cells) of the self-organizing map to train | 16 |
| `steps` | `int` | Training step count for the self-organizing map | 64 |
| `training_scale` | `float` | Nearest-neighbor scale image size prior to sampling - size close to sample size is recommended | 0.25 |
| `sample_width` | `int` | Width of assorted pixel sample per step - for performance, keep this number low | 64 |
| `sample_height` | `int` | Height of assorted pixel sample per step - for performance, keep this number low | 64 |


</details>

<details>
<summary>

#### Output

</summary>

**Type:** `ImageSOMOutput`

| Name | Type | Description |
| ---- | ---- | ----------- |
| `image_out` | `ImageField` | Quantized image |
| `map_out` | `ImageField` | The pixels of the self-organizing map |
| `image_width` | `int` | Width of the quantized image |
| `image_height` | `int` | Height of the quantized image |
| `map_width` | `int` | Width of the SOM image |
| `map_height` | `int` | Height of the SOM image |


</details>

---
### Image Search to Mask (Clipseg)
**ID:** `imgs2mask_clipseg`

**Category:** image

**Tags:** image, search, mask, clip, clipseg, imgs2mask

**Version:** 1.0.0

**Description:** Uses the Clipseg model to generate an image mask from an image prompt.

Input a base image and a prompt image to generate a mask representing areas of the base image matched by the prompt image contents.

<details>
<summary>

#### Inputs

</summary>

| Name | Type | Description | Default |
| ---- | ---- | ----------- | ------- |
| `image` | `ImageField` | The image from which to create a mask | None |
| `search_image` | `ImageField` | Image prompt for which to search | None |
| `invert_output` | `bool` | Off: white on black / On: black on white | True |
| `smoothing` | `float` | Radius of blur to apply before thresholding | 4.0 |
| `subject_threshold` | `float` | Threshold above which is considered the subject | 0.4 |
| `background_threshold` | `float` | Threshold below which is considered the background | 0.4 |
| `mask_expand_or_contract` | `int` | Pixels by which to grow (or shrink) mask after thresholding | 0 |
| `mask_blur` | `float` | Radius of blur to apply after thresholding | 0.0 |


</details>

<details>
<summary>

#### Output

</summary>

**Type:** `ImageOutput.build(...)`



</details>

---
### Latent Quantize (Kohonen map)
**ID:** `latent_som`

**Category:** latents

**Tags:** latents, quantize, som, kohonen

**Version:** 0.0.3

**Description:** Use a self-organizing map to quantize the values of a latent tensor.

This is highly experimental and not really suitable for most use cases. It's very easy to use settings that will appear to hang, or tie up the PC for a very long time, so use of this node is somewhat discouraged.

<details>
<summary>

#### Inputs

</summary>

| Name | Type | Description | Default |
| ---- | ---- | ----------- | ------- |
| `latents_in` | `LatentsField` | The latents tensor to quantize | None |
| `reference_in` | `Optional[LatentsField]` | Optional alternate latents to use for training | None |
| `width` | `int` | Width (in cells) of the self-organizing map | 4 |
| `height` | `int` | Height (in cells) of the self-organizing map | 3 |
| `steps` | `int` | Training step count for the self-organizing map | 256 |


</details>

<details>
<summary>

#### Output

</summary>

**Type:** `LatentsOutput.build(...)`



</details>

---
### Noise (Spectral characteristics)
**ID:** `noise_spectral`

**Category:** noise

**Tags:** noise

**Version:** 1.3.0

**Description:** Creates a latents tensor of 2D noise channels approximating the desired characteristics.

This operates like 2D Noise Image but outputs latent tensors, 4-channel or 16-channel.

<details>
<summary>

#### Inputs

</summary>

| Name | Type | Description | Default |
| ---- | ---- | ----------- | ------- |
| `noise_type` | `Literal[(White, Red, Blue, Green)]` | Desired noise spectral characteristics | White |
| `width` | `int` | Desired image width | 512 |
| `height` | `int` | Desired image height | 512 |
| `flux16` | `bool` | If false, 4-channel (SD/SDXL); if true, 16-channel (Flux) | False |
| `seed` | `int` | Seed for noise generation | 0 |
| `iterations` | `int` | Noise approx. iterations | 15 |
| `blur_threshold` | `float` | Threshold used in computing noise (lower is better/slower) | 0.2 |
| `sigma_red` | `float` | Sigma for strong gaussian blur LPF for red/green | 3.0 |
| `sigma_blue` | `float` | Sigma for weak gaussian blur HPF for blue/green | 1.0 |


</details>

<details>
<summary>

#### Output

</summary>

**Type:** `NoiseOutput.build(...)`



</details>

---
### Offset Image
**ID:** `offset_image`

**Category:** image

**Tags:** image, offset

**Version:** 1.2.0

**Description:** Offsets an image by a given percentage (or pixel amount).

This works like Offset Latents, but in image space, with the additional capability of taking exact pixel offsets instead of just percentages (toggled with a switch/boolean input).

<details>
<summary>

#### Inputs

</summary>

| Name | Type | Description | Default |
| ---- | ---- | ----------- | ------- |
| `as_pixels` | `bool` | Interpret offsets as pixels rather than percentages | False |
| `image` | `ImageField` | Image to be offset | None |
| `x_offset` | `float` | x-offset for the subject | 0.5 |
| `y_offset` | `float` | y-offset for the subject | 0.5 |


</details>

<details>
<summary>

#### Output

</summary>

**Type:** `ImageOutput.build(...)`



</details>

---
### Offset Latents
**ID:** `offset_latents`

**Category:** latents

**Tags:** latents, offset

**Version:** 1.2.0

**Description:** Offsets a latents tensor by a given percentage of height/width.

This takes a Latents input as well as two numbers (between 0 and 1), which are used to offset the latents in the vertical and/or horizontal directions. 0.5/0.5 would offset the image 50% in both directions such that the corners will wrap around and become the center of the image.

<details>
<summary>

#### Inputs

</summary>

| Name | Type | Description | Default |
| ---- | ---- | ----------- | ------- |
| `latents` | `LatentsField` |  | None |
| `x_offset` | `float` | Approx percentage to offset (H) | 0.5 |
| `y_offset` | `float` | Approx percentage to offset (V) | 0.5 |


</details>

<details>
<summary>

#### Output

</summary>

**Type:** `LatentsOutput.build(...)`



</details>

---
### RGB Merge
**ID:** `rgb_merge`

**Category:** image

**Tags:** rgb, image, color

**Version:** 1.0.0

**Description:** Merge RGB color channels and alpha

<details>
<summary>

#### Inputs

</summary>

| Name | Type | Description | Default |
| ---- | ---- | ----------- | ------- |
| `r_channel` | `ImageField` | The red channel | None |
| `g_channel` | `ImageField` | The green channel | None |
| `b_channel` | `ImageField` | The blue channel | None |
| `alpha_channel` | `ImageField` | The alpha channel | None |


</details>

<details>
<summary>

#### Output

</summary>

**Type:** `ImageOutput.build(...)`



</details>

---
### RGB Split
**ID:** `rgb_split`

**Category:** image

**Tags:** rgb, image, color

**Version:** 1.0.0

**Description:** Split an image into RGB color channels and alpha

<details>
<summary>

#### Inputs

</summary>

| Name | Type | Description | Default |
| ---- | ---- | ----------- | ------- |
| `image` | `ImageField` | The image to split into channels | None |


</details>

<details>
<summary>

#### Output

</summary>

**Type:** `RGBSplitOutput`

| Name | Type | Description |
| ---- | ---- | ----------- |
| `r_channel` | `ImageField` | Grayscale image of the red channel |
| `g_channel` | `ImageField` | Grayscale image of the green channel |
| `b_channel` | `ImageField` | Grayscale image of the blue channel |
| `alpha_channel` | `ImageField` | Grayscale image of the alpha channel |
| `width` | `int` | The width of the image in pixels |
| `height` | `int` | The height of the image in pixels |


</details>

---
### Rotate/Flip Image
**ID:** `rotate_image`

**Category:** image

**Tags:** image, rotate, flip

**Version:** 1.2.0

**Description:** Rotates an image by a given angle (in degrees clockwise).

Rotate an image in degrees about its center, clockwise (positive entries) or counterclockwise (negative entries). Optionally expand the image boundary to fit the rotated image, or flip it horizontally or vertically.

<details>
<summary>

#### Inputs

</summary>

| Name | Type | Description | Default |
| ---- | ---- | ----------- | ------- |
| `image` | `ImageField` | Image to be rotated clockwise | None |
| `degrees` | `float` | Angle (in degrees clockwise) by which to rotate | 90.0 |
| `expand_to_fit` | `bool` | If true, extends the image boundary to fit the rotated content | True |
| `flip_horizontal` | `bool` | If true, flips the image horizontally | False |
| `flip_vertical` | `bool` | If true, flips the image vertically | False |


</details>

<details>
<summary>

#### Output

</summary>

**Type:** `ImageOutput.build(...)`



</details>

---
### Shadows/Highlights/Midtones
**ID:** `shmmask`

**Category:** image

**Tags:** mask, image, shadows, highlights, midtones

**Version:** 1.2.0

**Description:** Extract a Shadows/Highlights/Midtones mask from an image.

Extract three masks (with adjustable hard or soft thresholds) representing shadows, midtones, and highlights regions of an image.

<details>
<summary>

#### Inputs

</summary>

| Name | Type | Description | Default |
| ---- | ---- | ----------- | ------- |
| `image` | `ImageField` | Image from which to extract mask | None |
| `invert_output` | `bool` | Off: white on black / On: black on white | True |
| `highlight_threshold` | `float` | Threshold beyond which mask values will be at extremum | 0.75 |
| `upper_mid_threshold` | `float` | Threshold to which to extend mask border by 0..1 gradient | 0.7 |
| `lower_mid_threshold` | `float` | Threshold to which to extend mask border by 0..1 gradient | 0.3 |
| `shadow_threshold` | `float` | Threshold beyond which mask values will be at extremum | 0.25 |
| `mask_expand_or_contract` | `int` | Pixels to grow (or shrink) the mask areas | 0 |
| `mask_blur` | `float` | Gaussian blur radius to apply to the masks | 0.0 |


</details>

<details>
<summary>

#### Output

</summary>

**Type:** `ShadowsHighlightsMidtonesMasksOutput`

| Name | Type | Description |
| ---- | ---- | ----------- |
| `highlights_mask` | `ImageField` | Soft-edged highlights mask |
| `midtones_mask` | `ImageField` | Soft-edged midtones mask |
| `shadows_mask` | `ImageField` | Soft-edged shadows mask |
| `width` | `int` | Width of the input/outputs |
| `height` | `int` | Height of the input/outputs |


</details>

---
### Text Mask
**ID:** `text_mask`

**Category:** mask

**Tags:** image, text, mask

**Version:** 1.2.0

**Description:** Creates a 2D rendering of a text mask from a given font.

Create a white on black (or black on white) text image for use with controlnets or further processing in other nodes. Specify any TTF/OTF font file available to Invoke and control parameters to resize, rotate, and reposition the text.

Currently this only generates one line of text, but it can be layered with other images using the Image Compositor node or any other such tool.

<details>
<summary>

#### Inputs

</summary>

| Name | Type | Description | Default |
| ---- | ---- | ----------- | ------- |
| `width` | `int` | The width of the desired mask | 512 |
| `height` | `int` | The height of the desired mask | 512 |
| `text` | `str` | The text to render |  |
| `font` | `str` | Path to a FreeType-supported TTF/OTF font file |  |
| `size` | `int` | Desired point size of text to use | 64 |
| `angle` | `float` | Angle of rotation to apply to the text | 0.0 |
| `x_offset` | `int` | x-offset for text rendering | 24 |
| `y_offset` | `int` | y-offset for text rendering | 36 |
| `invert` | `bool` | Whether to invert color of the output | False |


</details>

<details>
<summary>

#### Output

</summary>

**Type:** `ImageOutput.build(...)`



</details>

---
### Text to Mask (Clipseg)
**ID:** `txt2mask_clipseg`

**Category:** image

**Tags:** image, mask, clip, clipseg, txt2mask

**Version:** 1.2.1

**Description:** Uses the Clipseg model to generate an image mask from a text prompt.

Input a prompt and an image to generate a mask representing areas of the image matched by the prompt.

<details>
<summary>

#### Inputs

</summary>

| Name | Type | Description | Default |
| ---- | ---- | ----------- | ------- |
| `image` | `ImageField` | The image from which to create a mask | None |
| `invert_output` | `bool` | Off: white on black / On: black on white | True |
| `prompt` | `str` | The prompt with which to create a mask | None |
| `smoothing` | `float` | Radius of blur to apply before thresholding | 4.0 |
| `subject_threshold` | `float` | Threshold above which is considered the subject | 0.4 |
| `background_threshold` | `float` | Threshold below which is considered the background | 0.4 |
| `mask_expand_or_contract` | `int` | Pixels by which to grow (or shrink) mask after thresholding | 0 |
| `mask_blur` | `float` | Radius of blur to apply after thresholding | 0.0 |


</details>

<details>
<summary>

#### Output

</summary>

**Type:** `ImageOutput.build(...)`



</details>

---
### Text to Mask Advanced (Clipseg)
**ID:** `txt2mask_clipseg_adv`

**Category:** image

**Tags:** image, mask, clip, clipseg, txt2mask, advanced

**Version:** 1.2.2

**Description:** Uses the Clipseg model to generate an image mask from a text prompt.

Output up to four prompt masks combined with logical "and", logical "or", or as separate channels of an RGBA image.

<details>
<summary>

#### Inputs

</summary>

| Name | Type | Description | Default |
| ---- | ---- | ----------- | ------- |
| `image` | `ImageField` | The image from which to create a mask | None |
| `invert_output` | `bool` | Off: white on black / On: black on white | True |
| `prompt_1` | `str` | First prompt with which to create a mask | None |
| `prompt_2` | `str` | Second prompt with which to create a mask (optional) | None |
| `prompt_3` | `str` | Third prompt with which to create a mask (optional) | None |
| `prompt_4` | `str` | Fourth prompt with which to create a mask (optional) | None |
| `combine` | `Literal[tuple(COMBINE_MODES)]` | How to combine the results | None |
| `smoothing` | `float` | Radius of blur to apply before thresholding | 4.0 |
| `subject_threshold` | `float` | Threshold above which is considered the subject | 1.0 |
| `background_threshold` | `float` | Threshold below which is considered the background | 0.0 |


</details>

<details>
<summary>

#### Output

</summary>

**Type:** `ImageOutput.build(...)`



</details>

---

## Footnotes
### Nodes since moved to Invoke core:

The following nodes are no longer maintained here because they were
integrated with InvokeAI core. If you have a version prior to 5.4.2
and want to install them anyway, you can revert to the
pre-invoke-5_4 tag of this repo.

#### Adjust Image Hue Plus

Rotate the hue of an image in one of several different color spaces.

#### Blend Latents/Noise (Masked)

Use a mask to blend part of one latents/noise tensor into
another. Can be used to "renoise" sections during a multi-stage
[masked] denoising process.

#### Enhance Image

Boost or reduce color saturation, contrast, brightness, sharpness,
or invert colors of any image at any stage with this simple wrapper
for pillow [PIL]'s ImageEnhance module.

Color inversion is toggled with a simple switch, while each of the
four enhancer modes are activated by entering a value other than 1
in each corresponding input field. Values less than 1 will reduce
the corresponding property, while values greater than 1 will enhance
it.

#### Equivalent Achromatic Lightness

Calculates image lightness accounting for Helmholtz-Kohlrausch
effect based on a method described by High, Green, and Nussbaum
(2023) [https://doi.org/10.1002/col.22839].

#### Image Layer Blend

Perform a layered blend of two images using alpha
compositing. Opacity of top layer is selectable, and a mask image
may also be used. There are currently 23 blend modes supported and 8
color space modes. Four of the blend modes - Hue, Saturation, Color,
and Luminosity - are restricted to only 6 of the color space modes:
RGB and Linear RGB will convert to HSL for those blend
modes. Several of the other blend modes only operate on the
lightness channel of non-RGB color space modes.

Blend modes available: 
  Normal, Lighten Only, Darken Only, Lighten Only (EAL), Darken
  Only (EAL), Hue, Saturation, Color, Luminosity, Linear Dodge
  (Add), Subtract, Multiply, Divide, Screen, Overlay, Linear
  Burn, Difference, Hard Light, Soft Light, Vivid Light, Linear
  Light, Color Burn, Color Dodge

  Color space modes available:
    RGB, Linear RGB, HSL, HSV, Okhsl, Okhsv, Oklch (Oklab), LCh
    (CIELab)

#### Image Compositor

Take a subject from an image with a flat backdrop and layer it on
another image using a chroma key to specify a color value/threshold
to remove backdrop pixels, or leave the color blank and a "flood
select" will be used from the image corners.

The subject image may be scaled using the fill X and fill Y options
(enable both to stretch-fit).  Final subject position may also be
adjusted with X offset and Y offset. If used, chroma key may be
specified either as an (R, G, B) tuple, or a CSS-3 color string.

#### Image Dilate or Erode

Dilate or expand a mask (or any image!). This is equivalent to an
expand/contract operation.

#### Image Value Thresholds

Clip an image to pure black/white beyond specified thresholds.

