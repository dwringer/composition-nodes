from PIL import ImageChops

from invokeai.invocation_api import (
    BaseInvocation,
    ImageField,
    ImageOutput,
    InputField,
    InvocationContext,
    WithBoard,
    WithMetadata,
    invocation,
    ImageField,
    ImageOutput,
)


@invocation(
    "offset_image",
    title="Offset Image",
    tags=["image", "offset"],
    category="image",
    version="1.2.0",
)
class ImageOffsetInvocation(BaseInvocation, WithMetadata, WithBoard):
    """Offsets an image by a given percentage (or pixel amount)."""

    as_pixels: bool = InputField(default=False, description="Interpret offsets as pixels rather than percentages")
    image: ImageField = InputField(default=None, description="Image to be offset")
    x_offset: float = InputField(default=0.5, description="x-offset for the subject")
    y_offset: float = InputField(default=0.5, description="y-offset for the subject")

    def invoke(self, context: InvocationContext) -> ImageOutput:
        image_to_offset = context.images.get_pil(self.image.image_name).convert(mode="RGBA")

        # Scale and position the subject:
        x_offset, y_offset = self.x_offset, self.y_offset
        if self.as_pixels:
            x_offset = int(x_offset)
            y_offset = int(y_offset)
        else:
            x_offset = int(x_offset * image_to_offset.width)
            y_offset = int(y_offset * image_to_offset.height)
        if (self.x_offset != 0) or (self.y_offset != 0):
            image_to_offset = ImageChops.offset(image_to_offset, x_offset, yoffset=y_offset)

        image_dto = context.images.save(image_to_offset)

        return ImageOutput.build(image_dto)
