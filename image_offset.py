from ast import literal_eval as tuple_from_string
from functools import reduce

from PIL import Image, ImageOps, ImageChops, ImageDraw, ImageColor

from invokeai.app.models.image import ImageCategory, ResourceOrigin
from invokeai.app.invocations.baseinvocation import (
    BaseInvocation,
    InputField,
    invocation,
    InvocationContext,
    OutputField,
)

from invokeai.app.invocations.primitives import (
    ImageField,
    ImageOutput
)


@invocation(
    "offset_image",
    title="Offset Image",
    tags=["image", "offset"],
    category="image",
)
class ImageOffsetInvocation(BaseInvocation):
    """Offsets an image by a given percentage (or pixel amount)."""
    as_pixels: bool = InputField(
        default=False, description="Interpret offsets as pixels rather than percentages"
    )
    image: ImageField = InputField(
        default=None, description="Image to be offset"
    )
    x_offset:   float = InputField(default=0.5, description="x-offset for the subject")
    y_offset:   float = InputField(default=0.5, description="y-offset for the subject")

    def invoke(self, context: InvocationContext) -> ImageOutput:
        image_to_offset = context.services.images.get_pil_image(self.image.image_name).convert(mode="RGBA")

        # Scale and position the subject:
        x_offset, y_offset = self.x_offset, self.y_offset
        if self.as_pixels:
            x_offset = int(x_offset)
            y_offset = int(y_offset)
        else:
            x_offset = int(x_offset * image_to_offset.width)
            y_offset = int(y_offset * image_to_offset.height)
        if (self.x_offset != 0) or (self.y_offset != 0):
            image_to_offset = ImageChops.offset(
                image_to_offset, x_offset, yoffset=y_offset
            )

        image_dto = context.services.images.create(
            image=image_to_offset,
            image_origin=ResourceOrigin.INTERNAL,
            image_category=ImageCategory.GENERAL,
            node_id=self.id,
            session_id=context.graph_execution_state_id,
            is_intermediate=self.is_intermediate,
        )

        return ImageOutput(
            image=ImageField(image_name=image_dto.image_name),
            width=image_dto.width,
            height=image_dto.height,
        )
