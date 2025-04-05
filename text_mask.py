from PIL import Image, ImageDraw, ImageFont, ImageOps

from invokeai.app.invocations.fields import InputField, WithBoard, WithMetadata

if not hasattr(Image, "Resampling"):
    Image.Resampling = Image  # (Compatibilty for Pillow earlier than v9)

from invokeai.invocation_api import (
    BaseInvocation,
    ImageOutput,
    InvocationContext,
    invocation,
)


@invocation("text_mask", title="Text Mask", tags=["image", "text", "mask"], category="mask", version="1.2.0")
class TextMaskInvocation(BaseInvocation, WithMetadata, WithBoard):
    """Creates a 2D rendering of a text mask from a given font.

Create a white on black (or black on white) text image for use with controlnets or further processing in other nodes. Specify any TTF/OTF font file available to Invoke and control parameters to resize, rotate, and reposition the text.

Currently this only generates one line of text, but it can be layered with other images using the Image Compositor node or any other such tool.
"""

    width: int = InputField(default=512, description="The width of the desired mask")
    height: int = InputField(default=512, description="The height of the desired mask")
    text: str = InputField(default="", description="The text to render")
    font: str = InputField(default="", description="Path to a FreeType-supported TTF/OTF font file")
    size: int = InputField(default=64, description="Desired point size of text to use")
    angle: float = InputField(default=0.0, description="Angle of rotation to apply to the text")
    x_offset: int = InputField(default=24, description="x-offset for text rendering")
    y_offset: int = InputField(default=36, description="y-offset for text rendering")
    invert: bool = InputField(default=False, description="Whether to invert color of the output")

    def invoke(self, context: InvocationContext) -> ImageOutput:
        image_out = Image.new(mode="RGBA", size=(self.width, self.height), color=(0, 0, 0, 255))

        font = ImageFont.truetype(self.font, self.size)
        text_writer = ImageDraw.Draw(image_out)
        text_writer.text((self.x_offset, self.y_offset), self.text, font=font, fill=(255, 255, 255, 255))
        if self.angle != 0.0:
            image_out = image_out.rotate(
                self.angle,
                resample=Image.Resampling.BICUBIC,
                expand=False,
                center=(self.x_offset, self.y_offset),
                fillcolor=(0, 0, 0, 255),
            )
        if self.invert:
            image_out = ImageOps.invert(image_out.convert("RGB"))
        image_dto = context.images.save(image_out)
        return ImageOutput.build(image_dto)
