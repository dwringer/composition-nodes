from typing import Literal

import cv2
import numpy
from PIL import Image, ImageFilter
import torch
from torchvision.transforms.functional import to_pil_image as pil_image_from_tensor
from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation

from invokeai.app.invocations.baseinvocation import (
    BaseInvocation,
    InputField,
    InvocationContext,
    invocation,
)
from invokeai.app.invocations.primitives import (
    ImageField,
    ImageOutput,
)
from invokeai.app.models.image import ImageCategory, ResourceOrigin
from invokeai.backend.stable_diffusion.diffusers_pipeline import (
    image_resized_to_grid_as_tensor,
)
    

COMBINE_MODES: list = [
    "or",
    "and",
    "none (rgba multiplex)",
]

@invocation(
    "txt2mask_clipseg_adv",
    title="Text to Mask Advanced (Clipseg)",
    tags=["image", "mask", "clip", "clipseg", "txt2mask", "advanced"],
    category="image",
    version="1.0.3",
)
class TextToMaskClipsegAdvancedInvocation(BaseInvocation):
    """Uses the Clipseg model to generate an image mask from a text prompt"""

    image: ImageField = InputField(description="The image from which to create a mask")
    invert_output: bool = InputField(
        default=True, description="Off: white on black / On: black on white"
    )
    prompt_1: str = InputField(description="First prompt with which to create a mask")
    prompt_2: str = InputField(description="Second prompt with which to create a mask (optional)")
    prompt_3: str = InputField(description="Third prompt with which to create a mask (optional)")
    prompt_4: str = InputField(description="Fourth prompt with which to create a mask (optional)")
    combine: Literal[tuple(COMBINE_MODES)] = InputField(
        default=COMBINE_MODES[0], description="How to combine the results"
    )
    smoothing: float = InputField(
        default=4.0, description="Radius of blur to apply before thresholding"
    )
    subject_threshold: float = InputField(
        default=1.0, description="Threshold above which is considered the subject"
    )
    background_threshold: float = InputField(
        default=0.0, description="Threshold below which is considered the background"
    )


    def get_threshold_mask(self, image_tensor):
        img_tensor = image_tensor.clone()
        threshold_h, threshold_s = self.subject_threshold, self.background_threshold
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
                vmax, vmin = max(threshold_h, threshold_s), min(threshold_h, threshold_s)
                if (vmax == vmin):
                    img_tensor[mask] = vmin * ones_tensor[mask]
                elif self.invert_output:
                    img_tensor[mask] = torch.sub(1.0, (img_tensor[mask] - vmin) / (vmax - vmin))
                else:
                    img_tensor[mask] = (img_tensor[mask] - vmin) / (vmax - vmin)

        img_tensor[ones_mask] = ones_tensor[ones_mask]
        img_tensor[zeros_mask] = zeros_tensor[zeros_mask]

        return img_tensor


    def invoke(self, context: InvocationContext) -> ImageOutput:
        image_in = context.services.images.get_pil_image(self.image.image_name)
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
            combined = predictions[0,:,:]
            for i in range(predictions.shape[0] - 1):
                combined = torch.mul(combined, predictions[i+1,:,:])
            predictions = combined
            image_out = pil_image_from_tensor(predictions, mode="L")
                           
        elif combine_mode == "or":
            combined = torch.add(torch.mul(predictions[0,:,:], -1.), 1.)
            for i in range(predictions.shape[0] - 1):
                combined = torch.mul(combined, torch.add(torch.mul(predictions[i+1,:,:], -1.), 1.))
            predictions = torch.add(torch.mul(combined, -1.), 1.)
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

        image_dto = context.services.images.create(
            image=image_out,
            image_origin=ResourceOrigin.INTERNAL,
            image_category=ImageCategory.GENERAL,
            node_id=self.id,
            session_id=context.graph_execution_state_id,
            is_intermediate=self.is_intermediate,
            workflow=self.workflow,
        )
        return ImageOutput(
            image=ImageField(image_name=image_dto.image_name),
            width=image_dto.width,
            height=image_dto.height
        )


@invocation(
    "img_val_thresholds",
    title="Image Value Thresholds",
    tags=["image", "mask", "value", "threshold"],
    category="image",
    version="1.0.3"
)
class ImageValueThresholdsInvocation(BaseInvocation):
    """Clip image to pure black/white past specified thresholds"""
    image: ImageField = InputField(description="The image from which to create a mask")
    invert_output: bool = InputField(
        default=False, description="Make light areas dark and vice versa"
    )
    renormalize_values: bool = InputField(
        default=False, description="Rescale remaining values from minimum to maximum"
    )
    lightness_only: bool = InputField(default=False, description="If true, only applies to image lightness (CIELa*b*)")
    threshold_upper: float = InputField(
        default=0.5, description="Threshold above which will be set to full value"
    )
    threshold_lower: float = InputField(
        default=0.5, description="Threshold below which will be set to minimum value"
    )

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
                    if (vmax == vmin):
                        img_tensor[mask] = vmin * ones_tensor[mask]
                    elif self.invert_output:
                        img_tensor[mask] = torch.sub(1.0, (img_tensor[mask] - vmin) / (vmax - vmin))
                    else:
                        img_tensor[mask] = (img_tensor[mask] - vmin) / (vmax - vmin)

        img_tensor[ones_mask] = ones_tensor[ones_mask]
        img_tensor[zeros_mask] = zeros_tensor[zeros_mask]

        return img_tensor

    def invoke(self, context: InvocationContext) -> ImageOutput:
        image_in = context.services.images.get_pil_image(self.image.image_name)

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

            image_out = Image.merge(
                "LAB",
                (l_channel, image_out.getchannel("A"), image_out.getchannel("B"))
            )
            if (image_mode == "L") or (image_mode == "P"):
                image_out = image_out.convert("RGB")
            image_out = image_out.convert(image_mode)
            if "a" in image_in.mode.lower():
                image_out = Image.merge(
                    image_in.mode,
                    tuple([image_out.getchannel(c) for c in image_mode] + [alpha_channel])
                )
        else:
            image_out = image_resized_to_grid_as_tensor(image_in, normalize=False)
            image_out = self.get_threshold_mask(image_out)
            image_out = pil_image_from_tensor(image_out)
        
        image_dto = context.services.images.create(
            image=image_out,
            image_origin=ResourceOrigin.INTERNAL,
            image_category=ImageCategory.GENERAL,
            node_id=self.id,
            session_id=context.graph_execution_state_id,
            is_intermediate=self.is_intermediate,
            workflow=self.workflow,
        )
        return ImageOutput(
            image=ImageField(image_name=image_dto.image_name),
            width=image_dto.width,
            height=image_dto.height
        )


@invocation(
    "img_dilate_erode",
    title="Image Dilate or Erode",
    tags=["image", "mask", "dilate", "erode", "expand", "contract", "mask"],
    category="image",
    version="1.0.2",
)
class ImageDilateOrErodeInvocation(BaseInvocation):
    """Dilate (expand) or erode (contract) an image"""

    image: ImageField = InputField(description="The image from which to create a mask")
    lightness_only: bool = InputField(default=False, description="If true, only applies to image lightness (CIELa*b*)")
    radius: int = InputField(
        default=4, description="Pixels by which to dilate(expand) or erode (contract) the image"
    )


    def expand_or_contract(self, image_in):
        image_out = numpy.array(image_in)
        expand_radius = self.radius
        expand_fn = None
        if 0 < self.radius:
            expand_fn = cv2.dilate
        else:
            expand_radius *= -1
            expand_fn = cv2.erode
        image_out = expand_fn(
            image_out,
            numpy.uint8(
                numpy.array(
                    [[i**2 + j**2 for i in range(-expand_radius, expand_radius + 1)]
                     for j in range(-expand_radius, expand_radius + 1)]
                ) < (expand_radius + 1)**2,
            ),
            iterations=1
        )
        return Image.fromarray(image_out, mode=image_in.mode)
    

    def invoke(self, context: InvocationContext) -> ImageOutput:
        image_in = context.services.images.get_pil_image(self.image.image_name)
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
            image_out = Image.merge(
                "LAB",
                (l_channel, image_out.getchannel("A"), image_out.getchannel("B"))
            )
            if (image_mode == "L") or (image_mode == "P"):
                image_out = image_out.convert("RGB")
            image_out = image_out.convert(image_mode)
            if "a" in image_in.mode.lower():
                image_out = Image.merge(
                    image_in.mode,
                    tuple([image_out.getchannel(c) for c in image_mode] + [alpha_channel])
                )
        else:
            image_out = self.expand_or_contract(image_out)
        

        image_dto = context.services.images.create(
            image=image_out,
            image_origin=ResourceOrigin.INTERNAL,
            image_category=ImageCategory.GENERAL,
            node_id=self.id,
            session_id=context.graph_execution_state_id,
            is_intermediate=self.is_intermediate,
            workflow=self.workflow,
        )
        return ImageOutput(
            image=ImageField(image_name=image_dto.image_name),
            width=image_dto.width,
            height=image_dto.height
        )


