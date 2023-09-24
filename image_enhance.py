from typing import Literal, Optional

from PIL import Image, ImageOps, ImageEnhance, ImageDraw
from pydantic import Field

from invokeai.app.models.image import ImageCategory, ResourceOrigin
from invokeai.app.invocations.baseinvocation import (
    BaseInvocation,
    BaseInvocationOutput,
    invocation,
    invocation_output,
    InvocationContext,
    InputField,
)
from invokeai.app.invocations.primitives import (
    ImageField,
    ImageOutput
)

@invocation(
    "img_enhance",
    title="Enhance Image",
    tags=["enhance", "image"],
    category="image",
    version="1.0.0",    
)
class ImageEnhanceInvocation(BaseInvocation):
    """Applies processing from PIL's ImageEnhance module."""
    image:      ImageField = InputField(default=None, description="The image for which to apply processing")
    invert:     bool  = InputField(default=False, description="Whether to invert the image colors")
    color:      float = InputField(default=1.0, description="Color enhancement factor")
    contrast:   float = InputField(default=1.0, description="Contrast enhancement factor")
    brightness: float = InputField(default=1.0, description="Brightness enhancement factor")
    sharpness:  float = InputField(default=1.0, description="Sharpness enhancement factor")

    def invoke(self, context: InvocationContext) -> ImageOutput:
        image_out = context.services.images.get_pil_image(self.image.image_name)
        if self.invert:
            if not (image_out.mode in ("L", "RGB")):
                image_out = image_out.convert('RGB')
            image_out = ImageOps.invert(image_out)
        if self.color != 1.0:
            color_enhancer = ImageEnhance.Color(image_out)
            image_out = color_enhancer.enhance(self.color)
        if self.contrast != 1.0:
            contrast_enhancer = ImageEnhance.Contrast(image_out)
            image_out = contrast_enhancer.enhance(self.contrast)
        if self.brightness != 1.0:
            brightness_enhancer = ImageEnhance.Brightness(image_out)
            image_out = brightness_enhancer.enhance(self.brightness)
        if self.sharpness != 1.0:
            sharpness_enhancer = ImageEnhance.Sharpness(image_out)
            image_out = sharpness_enhancer.enhance(self.sharpness)
        image_dto = context.services.images.create(
            image=image_out,
            image_origin=ResourceOrigin.INTERNAL,
            image_category=ImageCategory.GENERAL,
            node_id=self.id,
            session_id=context.graph_execution_state_id,
            is_intermediate=self.is_intermediate
        )
        return ImageOutput(image=ImageField(image_name=image_dto.image_name),
                           width=image_dto.width,
                           height=image_dto.height,
        )
