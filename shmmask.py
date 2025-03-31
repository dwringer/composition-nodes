from math import pi as PI

import cv2
import numpy
import PIL.Image
import torch
from torchvision.transforms.functional import to_pil_image as pil_image_from_tensor

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


@invocation_output("shmmask_output")
class ShadowsHighlightsMidtonesMasksOutput(BaseInvocationOutput):
    highlights_mask: ImageField = OutputField(default=None, description="Soft-edged highlights mask")
    midtones_mask: ImageField = OutputField(default=None, description="Soft-edged midtones mask")
    shadows_mask: ImageField = OutputField(default=None, description="Soft-edged shadows mask")
    width: int = OutputField(description="Width of the input/outputs")
    height: int = OutputField(description="Height of the input/outputs")


@invocation(
    "shmmask",
    title="Shadows/Highlights/Midtones",
    tags=["mask", "image", "shadows", "highlights", "midtones"],
    category="image",
    version="1.2.0",
)
class ShadowsHighlightsMidtonesMaskInvocation(BaseInvocation, WithMetadata, WithBoard):
    """Extract a Shadows/Highlights/Midtones mask from an image"""

    image: ImageField = InputField(description="Image from which to extract mask")
    invert_output: bool = InputField(default=True, description="Off: white on black / On: black on white")
    highlight_threshold: float = InputField(
        default=0.75, description="Threshold beyond which mask values will be at extremum"
    )
    upper_mid_threshold: float = InputField(
        default=0.7, description="Threshold to which to extend mask border by 0..1 gradient"
    )
    lower_mid_threshold: float = InputField(
        default=0.3, description="Threshold to which to extend mask border by 0..1 gradient"
    )
    shadow_threshold: float = InputField(
        default=0.25, description="Threshold beyond which mask values will be at extremum"
    )
    mask_expand_or_contract: int = InputField(default=0, description="Pixels to grow (or shrink) the mask areas")
    mask_blur: float = InputField(default=0.0, description="Gaussian blur radius to apply to the masks")

    def expand_or_contract(self, image_in):
        image_out = numpy.array(image_in)
        expand_radius = self.mask_expand_or_contract
        expand_fn = None
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (abs(expand_radius * 2), abs(expand_radius * 2)))
        if 0 < self.mask_expand_or_contract:
            if self.invert_output:
                expand_fn = cv2.erode
            else:
                expand_fn = cv2.dilate
        else:
            if self.invert_output:
                expand_fn = cv2.dilate
            else:
                expand_fn = cv2.erode
        image_out = expand_fn(image_out, kernel, iterations=1)
        return PIL.Image.fromarray(image_out, mode="L")

    def get_highlights_mask(self, image_tensor):
        img_tensor = image_tensor.clone()
        threshold_h, threshold_s = self.highlight_threshold, self.upper_mid_threshold
        ones_tensor = torch.ones(img_tensor.shape)
        zeros_tensor = torch.zeros(img_tensor.shape)

        zeros_mask = torch.ge(img_tensor, threshold_h)
        ones_mask = torch.lt(img_tensor, threshold_s)
        if not (threshold_h == threshold_s):
            mask_hi = torch.ge(img_tensor, threshold_s)
            mask_lo = torch.lt(img_tensor, threshold_h)
            mask = torch.logical_and(mask_hi, mask_lo)
            masked = img_tensor[mask]
            if 0 < masked.numel():
                vmax, vmin = max(threshold_h, threshold_s), min(threshold_h, threshold_s)
                if vmax == vmin:
                    img_tensor[mask] = vmin * ones_tensor
                else:
                    img_tensor[mask] = torch.sub(1.0, (img_tensor[mask] - vmin) / (vmax - vmin))  # hi is 0

        img_tensor[ones_mask] = ones_tensor[ones_mask]
        img_tensor[zeros_mask] = zeros_tensor[zeros_mask]

        return img_tensor if self.invert_output else torch.sub(1.0, img_tensor)

    def get_shadows_mask(self, image_tensor):
        img_tensor = image_tensor.clone()
        threshold_h, threshold_s = self.shadow_threshold, self.lower_mid_threshold
        ones_tensor = torch.ones(img_tensor.shape)
        zeros_tensor = torch.zeros(img_tensor.shape)

        zeros_mask = torch.le(img_tensor, threshold_h)
        ones_mask = torch.gt(img_tensor, threshold_s)
        if not (threshold_h == threshold_s):
            mask_hi = torch.le(img_tensor, threshold_s)
            mask_lo = torch.gt(img_tensor, threshold_h)
            mask = torch.logical_and(mask_hi, mask_lo)
            masked = img_tensor[mask]
            if 0 < masked.numel():
                vmax, vmin = max(threshold_h, threshold_s), min(threshold_h, threshold_s)
                if vmax == vmin:
                    img_tensor[mask] = vmin * ones_tensor
                else:
                    img_tensor[mask] = (img_tensor[mask] - vmin) / (vmax - vmin)  # lo is 0

        img_tensor[ones_mask] = ones_tensor[ones_mask]
        img_tensor[zeros_mask] = zeros_tensor[zeros_mask]

        return img_tensor if self.invert_output else torch.sub(1.0, img_tensor)

    def get_midtones_mask(self, image_tensor):
        img_tensor = image_tensor.clone()
        h_threshold_hard, h_threshold_soft = self.highlight_threshold, self.upper_mid_threshold
        s_threshold_hard, s_threshold_soft = self.shadow_threshold, self.lower_mid_threshold
        ones_tensor = torch.ones(img_tensor.shape)
        zeros_tensor = torch.zeros(img_tensor.shape)

        mask_lo = torch.le(img_tensor, h_threshold_soft)
        mask_hi = torch.ge(img_tensor, s_threshold_soft)
        mid_mask = torch.logical_and(mask_hi, mask_lo)
        highlight_ones_mask = torch.gt(img_tensor, h_threshold_hard)
        shadows_ones_mask = torch.lt(img_tensor, s_threshold_hard)
        mask_top_hi = torch.gt(img_tensor, h_threshold_soft)
        mask_top_lo = torch.le(img_tensor, h_threshold_hard)
        mask_top = torch.logical_and(mask_top_hi, mask_top_lo)
        mask_bottom_hi = torch.ge(img_tensor, s_threshold_hard)
        mask_bottom_lo = torch.lt(img_tensor, s_threshold_soft)
        mask_bottom = torch.logical_and(mask_bottom_hi, mask_bottom_lo)

        if not (h_threshold_hard == h_threshold_soft):
            masked = img_tensor[mask_top]
            if 0 < masked.numel():
                vmax_top, vmin_top = (max(h_threshold_hard, h_threshold_soft), min(h_threshold_hard, h_threshold_soft))
                if vmax_top == vmin_top:
                    img_tensor[mask_top] = vmin_top * ones_tensor
                else:
                    img_tensor[mask_top] = (img_tensor[mask_top] - vmin_top) / (vmax_top - vmin_top)  # hi is 1

        if not (s_threshold_hard == s_threshold_soft):
            masked = img_tensor[mask_bottom]
            if 0 < masked.numel():
                vmax_bottom, vmin_bottom = (
                    max(s_threshold_hard, s_threshold_soft),
                    min(s_threshold_hard, s_threshold_soft),
                )
                if vmax_bottom == vmin_bottom:
                    img_tensor[mask_bottom] = vmin_bottom * ones_tensor
                else:
                    img_tensor[mask_bottom] = torch.sub(
                        1.0, (img_tensor[mask_bottom] - vmin_bottom) / (vmax_bottom - vmin_bottom)
                    )  # lo is 1

        img_tensor[mid_mask] = zeros_tensor[mid_mask]
        img_tensor[highlight_ones_mask] = ones_tensor[highlight_ones_mask]
        img_tensor[shadows_ones_mask] = ones_tensor[shadows_ones_mask]

        return img_tensor if self.invert_output else torch.sub(1.0, img_tensor)

    def invoke(self, context: InvocationContext) -> ShadowsHighlightsMidtonesMasksOutput:
        image_in = context.images.get_pil(self.image.image_name)
        if image_in.mode != "L":
            image_in = image_in.convert("L")
        image_tensor = image_resized_to_grid_as_tensor(image_in, normalize=False, multiple_of=1)
        h_image_out = pil_image_from_tensor(self.get_highlights_mask(image_tensor), mode="L")
        if self.mask_expand_or_contract != 0:
            h_image_out = self.expand_or_contract(h_image_out)

        if 0 < self.mask_blur:
            h_image_out = h_image_out.filter(PIL.ImageFilter.GaussianBlur(radius=self.mask_blur))
        h_image_dto = context.images.save(h_image_out)
        m_image_out = pil_image_from_tensor(self.get_midtones_mask(image_tensor), mode="L")
        if self.mask_expand_or_contract != 0:
            m_image_out = self.expand_or_contract(m_image_out)

        if 0 < self.mask_blur:
            m_image_out = m_image_out.filter(PIL.ImageFilter.GaussianBlur(radius=self.mask_blur))
        m_image_dto = context.images.save(m_image_out)
        s_image_out = pil_image_from_tensor(self.get_shadows_mask(image_tensor), mode="L")
        if self.mask_expand_or_contract != 0:
            s_image_out = self.expand_or_contract(s_image_out)
        if 0 < self.mask_blur:
            s_image_out = s_image_out.filter(PIL.ImageFilter.GaussianBlur(radius=self.mask_blur))
        s_image_dto = context.images.save(s_image_out)
        return ShadowsHighlightsMidtonesMasksOutput(
            highlights_mask=ImageField(image_name=h_image_dto.image_name),
            midtones_mask=ImageField(image_name=m_image_dto.image_name),
            shadows_mask=ImageField(image_name=s_image_dto.image_name),
            width=h_image_dto.width,
            height=h_image_dto.height,
        )
