from typing import Literal

import cv2
import numpy
import torch
from PIL import Image, ImageFilter
from torchvision.transforms.functional import to_pil_image as pil_image_from_tensor
from transformers import CLIPSegForImageSegmentation, CLIPSegProcessor

from invokeai.backend.stable_diffusion.diffusers_pipeline import image_resized_to_grid_as_tensor
from invokeai.invocation_api import (
    BaseInvocation,
    BaseInvocationOutput,
    ImageField,
    ImageOutput,
    InputField,
    InvocationContext,
    OutputField,
    WithBoard,
    WithMetadata,
    invocation,
    invocation_output,
)

from .clipseg import ClipsegBase

COMBINE_MODES: list = [
    "or",
    "and",
    "butnot",
    "none (rgba multiplex)",
]


@invocation(
    "txt2mask_clipseg_adv",
    title="Text to Mask Advanced (Clipseg)",
    tags=["image", "mask", "clip", "clipseg", "txt2mask", "advanced"],
    category="image",
    version="1.2.2",
)
class TextToMaskClipsegAdvancedInvocation(BaseInvocation, ClipsegBase, WithMetadata, WithBoard):
    """Uses the Clipseg model to generate an image mask from a text prompt"""

    image: ImageField = InputField(description="The image from which to create a mask")
    invert_output: bool = InputField(default=True, description="Off: white on black / On: black on white")
    prompt_1: str = InputField(description="First prompt with which to create a mask")
    prompt_2: str = InputField(description="Second prompt with which to create a mask (optional)")
    prompt_3: str = InputField(description="Third prompt with which to create a mask (optional)")
    prompt_4: str = InputField(description="Fourth prompt with which to create a mask (optional)")
    combine: Literal[tuple(COMBINE_MODES)] = InputField(
        default=COMBINE_MODES[0], description="How to combine the results"
    )
    smoothing: float = InputField(default=4.0, description="Radius of blur to apply before thresholding")
    subject_threshold: float = InputField(default=1.0, description="Threshold above which is considered the subject")
    background_threshold: float = InputField(
        default=0.0, description="Threshold below which is considered the background"
    )

    def invoke(self, context: InvocationContext) -> ImageOutput:
        image_in = context.images.get_pil(self.image.image_name)
        image_size = image_in.size
        image_out = None

        processor = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")

        model = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined")

        image_in = image_in.convert("RGB")

        prompts = [self.prompt_1]
        for prompt in [self.prompt_2, self.prompt_3, self.prompt_4]:
            if 0 < len(prompt.strip()):
                prompts.append(prompt)

        input_args = processor(
            text=prompts, images=[image_in for p in prompts], padding="max_length", return_tensors="pt"
        )

        with torch.no_grad():
            output = model(**input_args)

        predictions = output.logits
        if len(predictions.shape) == 2:
            predictions = predictions.unsqueeze(0)
        predictions = torch.sigmoid(predictions)

        combine_mode = self.combine.split()[0]
        if combine_mode == "and":
            combined = predictions[0, :, :]
            for i in range(predictions.shape[0] - 1):
                combined = torch.mul(combined, predictions[i + 1, :, :])
            predictions = combined
            image_out = pil_image_from_tensor(predictions, mode="L")

        elif combine_mode == "or":
            combined = torch.add(torch.mul(predictions[0, :, :], -1.0), 1.0)
            for i in range(predictions.shape[0] - 1):
                combined = torch.mul(combined, torch.add(torch.mul(predictions[i + 1, :, :], -1.0), 1.0))
            predictions = torch.add(torch.mul(combined, -1.0), 1.0)
            image_out = pil_image_from_tensor(predictions, mode="L")

        elif combine_mode == "butnot":
            combined = predictions[0, :, :]
            for i in range(predictions.shape[0] - 1):
                combined = torch.mul(
                    combined,
                    torch.sub(1.0, predictions[i + 1, :, :])
                )
            predictions = combined
            image_out = pil_image_from_tensor(predictions, mode="L")

        else:
            missing_count = 4 - predictions.shape[0]
            extras = torch.ones([missing_count] + list(predictions.shape[1:]))
            predictions = torch.cat([predictions, extras], 0)
            image_out = pil_image_from_tensor(predictions, mode="RGBA")

        image_out = image_out.resize(image_size)

        image_out = image_resized_to_grid_as_tensor(image_out, normalize=False)
        image_out = (image_out - image_out.min()) / (image_out.max() - image_out.min())
        image_out = pil_image_from_tensor(image_out)

        if 0 < self.smoothing:
            image_out = image_out.filter(ImageFilter.GaussianBlur(radius=self.smoothing))

        image_out = image_resized_to_grid_as_tensor(image_out, normalize=False)
        image_out = self.get_threshold_mask(image_out)
        image_out = pil_image_from_tensor(image_out)

        image_dto = context.images.save(image_out)

        return ImageOutput.build(image_dto)


@invocation(
    "img_val_thresholds",
    title="Image Value Thresholds",
    tags=["image", "mask", "value", "threshold"],
    category="image",
    version="1.2.0",
)
class ImageValueThresholdsInvocation(BaseInvocation, WithMetadata, WithBoard):
    """Clip image to pure black/white past specified thresholds"""

    image: ImageField = InputField(description="The image from which to create a mask")
    invert_output: bool = InputField(default=False, description="Make light areas dark and vice versa")
    renormalize_values: bool = InputField(default=False, description="Rescale remaining values from minimum to maximum")
    lightness_only: bool = InputField(default=False, description="If true, only applies to image lightness (CIELa*b*)")
    threshold_upper: float = InputField(default=0.5, description="Threshold above which will be set to full value")
    threshold_lower: float = InputField(default=0.5, description="Threshold below which will be set to minimum value")

    def get_threshold_mask(self, image_tensor):
        img_tensor = image_tensor.clone()
        threshold_h, threshold_s = self.threshold_upper, self.threshold_lower
        ones_tensor = torch.ones(img_tensor.shape)
        zeros_tensor = torch.zeros(img_tensor.shape)

        zeros_mask, ones_mask = None, None
        if self.invert_output:
            zeros_mask, ones_mask = torch.ge(img_tensor, threshold_h), torch.lt(img_tensor, threshold_s)
        else:
            ones_mask, zeros_mask = torch.ge(img_tensor, threshold_h), torch.lt(img_tensor, threshold_s)

        if not (threshold_h == threshold_s):
            mask_hi = torch.ge(img_tensor, threshold_s)
            mask_lo = torch.lt(img_tensor, threshold_h)
            mask = torch.logical_and(mask_hi, mask_lo)
            masked = img_tensor[mask]
            if 0 < masked.numel():
                if self.renormalize_values:
                    vmax, vmin = max(threshold_h, threshold_s), min(threshold_h, threshold_s)
                    if vmax == vmin:
                        img_tensor[mask] = vmin * ones_tensor[mask]
                    elif self.invert_output:
                        img_tensor[mask] = torch.sub(1.0, (img_tensor[mask] - vmin) / (vmax - vmin))
                    else:
                        img_tensor[mask] = (img_tensor[mask] - vmin) / (vmax - vmin)

        img_tensor[ones_mask] = ones_tensor[ones_mask]
        img_tensor[zeros_mask] = zeros_tensor[zeros_mask]

        return img_tensor

    def invoke(self, context: InvocationContext) -> ImageOutput:
        image_in = context.images.get_pil(self.image.image_name)

        if self.lightness_only:
            image_mode = image_in.mode
            alpha_channel = None
            if (image_mode == "RGBA") or (image_mode == "LA") or (image_mode == "PA"):
                alpha_channel = image_in.getchannel("A")
            elif (image_mode == "RGBa") or (image_mode == "La") or (image_mode == "Pa"):
                alpha_channel = image_in.getchannel("a")
            if (image_mode == "RGBA") or (image_mode == "RGBa"):
                image_mode = "RGB"
            elif (image_mode == "LA") or (image_mode == "La"):
                image_mode = "L"
            elif image_mode == "PA":
                image_mode = "P"
            image_out = image_in.convert("RGB")
            image_out = image_out.convert("LAB")

            l_channel = image_resized_to_grid_as_tensor(image_out.getchannel("L"), normalize=False)
            l_channel = self.get_threshold_mask(l_channel)
            l_channel = pil_image_from_tensor(l_channel)

            image_out = Image.merge("LAB", (l_channel, image_out.getchannel("A"), image_out.getchannel("B")))
            if (image_mode == "L") or (image_mode == "P"):
                image_out = image_out.convert("RGB")
            image_out = image_out.convert(image_mode)
            if "a" in image_in.mode.lower():
                image_out = Image.merge(
                    image_in.mode, tuple([image_out.getchannel(c) for c in image_mode] + [alpha_channel])
                )
        else:
            image_out = image_resized_to_grid_as_tensor(image_in, normalize=False)
            image_out = self.get_threshold_mask(image_out)
            image_out = pil_image_from_tensor(image_out)

        image_dto = context.images.save(image_out)

        return ImageOutput.build(image_dto)


DILATE_ERODE_MODES: list = [
    "Dilate",
    "Erode",
]


@invocation(
    "img_dilate_erode",
    title="Image Dilate or Erode",
    tags=["image", "mask", "dilate", "erode", "expand", "contract", "mask"],
    category="image",
    version="1.3.0",
)
class ImageDilateOrErodeInvocation(BaseInvocation, WithMetadata):
    """Dilate (expand) or erode (contract) an image"""

    image: ImageField = InputField(description="The image from which to create a mask")
    lightness_only: bool = InputField(default=False, description="If true, only applies to image lightness (CIELa*b*)")
    radius_w: int = InputField(
        ge=0, default=4, description="Width (in pixels) by which to dilate(expand) or erode (contract) the image"
    )
    radius_h: int = InputField(
        ge=0, default=4, description="Height (in pixels) by which to dilate(expand) or erode (contract) the image"
    )
    mode: Literal[tuple(DILATE_ERODE_MODES)] = InputField(
        default=DILATE_ERODE_MODES[0], description="How to operate on the image"
    )

    def expand_or_contract(self, image_in):
        image_out = numpy.array(image_in)
        expand_radius_w = self.radius_w
        expand_radius_h = self.radius_h

        expand_fn = None
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (expand_radius_w * 2 + 1, expand_radius_h * 2 + 1))
        if self.mode == "Dilate":
            expand_fn = cv2.dilate
        elif self.mode == "Erode":
            expand_fn = cv2.erode
        image_out = expand_fn(image_out, kernel, iterations=1)
        return Image.fromarray(image_out, mode=image_in.mode)

    def invoke(self, context: InvocationContext) -> ImageOutput:
        image_in = context.images.get_pil(self.image.image_name)
        image_out = image_in

        if self.lightness_only:
            image_mode = image_in.mode
            alpha_channel = None
            if (image_mode == "RGBA") or (image_mode == "LA") or (image_mode == "PA"):
                alpha_channel = image_in.getchannel("A")
            elif (image_mode == "RGBa") or (image_mode == "La") or (image_mode == "Pa"):
                alpha_channel = image_in.getchannel("a")
            if (image_mode == "RGBA") or (image_mode == "RGBa"):
                image_mode = "RGB"
            elif (image_mode == "LA") or (image_mode == "La"):
                image_mode = "L"
            elif image_mode == "PA":
                image_mode = "P"
            image_out = image_out.convert("RGB")
            image_out = image_out.convert("LAB")
            l_channel = self.expand_or_contract(image_out.getchannel("L"))
            image_out = Image.merge("LAB", (l_channel, image_out.getchannel("A"), image_out.getchannel("B")))
            if (image_mode == "L") or (image_mode == "P"):
                image_out = image_out.convert("RGB")
            image_out = image_out.convert(image_mode)
            if "a" in image_in.mode.lower():
                image_out = Image.merge(
                    image_in.mode, tuple([image_out.getchannel(c) for c in image_mode] + [alpha_channel])
                )
        else:
            image_out = self.expand_or_contract(image_out)

        image_dto = context.images.save(image_out)

        return ImageOutput.build(image_dto)

@invocation_output("clipseg_mask_hierarchy_output")
class ClipsegMaskHierarchyOutput(BaseInvocationOutput):
    """Class for invocations that output a hierarchy of masks"""
    mask_1: ImageField = OutputField(
        default=None,
        description="Mask corresponding to prompt 1 (full coverage)"
    )
    mask_2: ImageField = OutputField(
        default=None,
        description="Mask corresponding to prompt 2 (minus mask 1)"
    )
    mask_3: ImageField = OutputField(
        default=None,
        description="Mask corresponding to prompt 3 (minus masks 1 & 2)"
    )
    mask_4: ImageField = OutputField(
        default=None,
        description="Mask corresponding to prompt 4 (minus masks 1, 2, & 3)"
    )
    mask_5: ImageField = OutputField(
        default=None,
        description="Mask corresponding to prompt 5 (minus masks 1 thru 4)"
    )
    mask_6: ImageField = OutputField(
        default=None,
        description="Mask corresponding to prompt 6 (minus masks 1 thru 5)"
    )
    mask_7: ImageField = OutputField(
        default=None,
        description="Mask corresponding to prompt 7 (minus masks 1 thru 6)"
    )
    ground_mask: ImageField = OutputField(
        default=None,
        description="Mask coresponding to remaining unmatched image areas."
    )


@invocation(
    "clipseg_mask_hierarchy",
    title="Clipseg Mask Hierarchy",
    tags=["image", "mask", "clip", "clipseg", "txt2mask", "hierarchy"],
    category="image",
    version="1.2.2",
)
class ClipsegMaskHierarchyInvocation(BaseInvocation, ClipsegBase, WithMetadata, WithBoard):
    """Creates a segmentation hierarchy of mutually exclusive masks from clipseg text prompts"""

    image: ImageField = InputField(description="The image from which to create masks")
    invert_output: bool = InputField(default=True, description="Off: white on black / On: black on white")
    smoothing: float = InputField(default=4.0, description="Radius of blur to apply before thresholding")
    prompt_1: str = InputField(description="Text to mask prompt with highest segmentation priority")
    threshold_1: float = InputField(default=0.4, description="Detection confidence threshold for prompt 1")
    prompt_2: str = InputField(description="Text to mask prompt, behind prompt 1")
    threshold_2: float = InputField(default=0.4, description="Detection confidence threshold for prompt 2")
    prompt_3: str = InputField(description="Text to mask prompt, behind prompts 1 & 2")
    threshold_3: float = InputField(default=0.4, description="Detection confidence threshold for prompt 3")
    prompt_4: str = InputField(description="Text to mask prompt, behind prompts 1, 2, & 3")
    threshold_4: float = InputField(default=0.4, description="Detection confidence threshold for prompt 4")
    prompt_5: str = InputField(description="Text to mask prompt, behind prompts 1 thru 4")
    threshold_5: float = InputField(default=0.4, description="Detection confidence threshold for prompt 5")
    prompt_6: str = InputField(description="Text to mask prompt, behind prompts 1 thru 5")
    threshold_6: float = InputField(default=0.4, description="Detection confidence threshold for prompt 6")
    prompt_7: str = InputField(description="Text to mask prompt, lowest priority behind all others")
    threshold_7: float = InputField(default=0.4, description="Detection confidence threshold for prompt 7")

    def invoke(self, context: InvocationContext) -> ClipsegMaskHierarchyOutput:

        image_in = context.images.get_pil(self.image.image_name)
        image_size = image_in.size

        processor = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")

        model = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined")

        image_in = image_in.convert("RGB")
        all_prompts = [
            self.prompt_1,
            self.prompt_2,
            self.prompt_3,
            self.prompt_4,
            self.prompt_5,
            self.prompt_6,
            self.prompt_7,
        ]
        all_thresholds = [
            self.threshold_1,
            self.threshold_2,
            self.threshold_3,
            self.threshold_4,
            self.threshold_5,
            self.threshold_6,
            self.threshold_7,
        ]

        prompts = []
        for prompt in all_prompts:
            if 0 < len(prompt.strip()):
                prompts.append(prompt)

        input_args = processor(
            text=prompts, images=[image_in for p in prompts], padding="max_length", return_tensors="pt"
        )

        with torch.no_grad():
            output = model(**input_args)

        predictions = output.logits
        if len(predictions.shape) == 2:
            predictions = predictions.unsqueeze(0)
        predictions = torch.sigmoid(predictions)

        masks = []
        ones, zeros = torch.ones(predictions[0].shape), torch.zeros(predictions[0].shape)
        ground_mask = torch.ones(predictions[0].shape)
        for i in range(7):
            if len(all_prompts[i].strip()) == 0:
                masks.append(zeros)
            else:
                mask = predictions[i, :, :]
                if 0 < self.smoothing:
                    mask = pil_image_from_tensor(mask)
                    mask = mask.filter(ImageFilter.GaussianBlur(radius=self.smoothing))
                    mask = image_resized_to_grid_as_tensor(mask, normalize=False)
                mask = torch.where(
                    all_thresholds[i] < mask,
                    ones,
                    zeros
                )
                ground_mask = torch.mul(
                    ground_mask,
                    torch.sub(1.0, mask)
                )
                for prev_mask in masks[0:i]:
                    mask = torch.mul(mask, torch.sub(1.0, prev_mask))

                masks.append(mask)

        masks_out = []
        for mask in masks:
            mask = pil_image_from_tensor(mask if not self.invert_output else torch.sub(1.0, mask), mode="L")
            mask_dto = context.images.save(mask)
            masks_out.append(ImageField(image_name=mask_dto.image_name))
        ground_mask = pil_image_from_tensor(
            ground_mask if not self.invert_output else torch.sub(1.0, ground_mask), mode="L"
        )
        ground_mask_dto = context.images.save(ground_mask)
        ground_mask_out = ImageField(image_name=ground_mask_dto.image_name)

        return ClipsegMaskHierarchyOutput(
            mask_1=masks_out[0],
            mask_2=masks_out[1],
            mask_3=masks_out[2],
            mask_4=masks_out[3],
            mask_5=masks_out[4],
            mask_6=masks_out[5],
            mask_7=masks_out[6],
            ground_mask=ground_mask_out
        )
