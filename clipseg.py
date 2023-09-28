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
    

@invocation(
    "txt2mask_clipseg",
    title="Text to Mask (Clipseg)",
    tags=["image", "mask", "clip", "clipseg", "txt2mask"],
    category="image",
    version="1.0.3",
)
class TextToMaskClipsegInvocation(BaseInvocation):
    """Uses the Clipseg model to generate an image mask from a text prompt"""

    image: ImageField = InputField(description="The image from which to create a mask")
    invert_output: bool = InputField(
        default=True, description="Off: white on black / On: black on white"
    )
    prompt: str = InputField(description="The prompt with which to create a mask")
    smoothing: float = InputField(
        default=4.0, description="Radius of blur to apply before thresholding"
    )
    subject_threshold: float = InputField(
        default=0.4, description="Threshold above which is considered the subject"
    )
    background_threshold: float = InputField(
        default=0.4, description="Threshold below which is considered the background"
    )
    mask_expand_or_contract: int = InputField(
        default=0, description="Pixels by which to grow (or shrink) mask after thresholding"
    )
    mask_blur: float = InputField(
        default=0.0, description="Radius of blur to apply after thresholding"
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


    def expand_or_contract(self, image_in):
        image_out = numpy.array(image_in)
        expand_radius = self.mask_expand_or_contract
        expand_fn = None
        if 0 < self.mask_expand_or_contract:
            if self.invert_output:
                expand_fn = cv2.erode
            else:
                expand_fn = cv2.dilate
        else:
            expand_radius *= -1
            if self.invert_output:
                expand_fn = cv2.dilate
            else:
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
        return Image.fromarray(image_out, mode="L")
    

    def invoke(self, context: InvocationContext) -> ImageOutput:
        image_in = context.services.images.get_pil_image(self.image.image_name)
        image_size = image_in.size
        image_out = None

        processor = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")

        model = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined")

        image_in = image_in.convert("RGB")
        
        input_args = processor(
            text=[self.prompt], images=[image_in], padding="max_length", return_tensors="pt"
        )

        with torch.no_grad():
            output = model(**input_args)
            
        predictions = output.logits

        image_out = pil_image_from_tensor(torch.sigmoid(predictions), mode="L")
        image_out = image_out.resize(image_size)

        image_out = image_resized_to_grid_as_tensor(image_out, normalize=False)
        image_out = (image_out - image_out.min()) / (image_out.max() - image_out.min())
        image_out = pil_image_from_tensor(image_out)

        if 0 < self.smoothing:
            image_out = image_out.filter(ImageFilter.GaussianBlur(radius=self.smoothing))

        image_out = image_resized_to_grid_as_tensor(image_out, normalize=False)
        image_out = self.get_threshold_mask(image_out)
        image_out = pil_image_from_tensor(image_out)

        if self.mask_expand_or_contract != 0:
            image_out = self.expand_or_contract(image_out)
        
        if 0 < self.mask_blur:
            image_out = image_out.filter(ImageFilter.GaussianBlur(radius=self.mask_blur))

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
