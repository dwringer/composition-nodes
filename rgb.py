import numpy
from PIL import Image
from typing import Optional

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

@invocation_output("rgb_split_output")
class RGBSplitOutput(BaseInvocationOutput):
    """Base class for invocations that output three L-mode images (R, G, B)"""

    r_channel: ImageField = OutputField(description="Grayscale image of the red channel")
    g_channel: ImageField = OutputField(description="Grayscale image of the green channel")
    b_channel: ImageField = OutputField(description="Grayscale image of the blue channel")
    alpha_channel: ImageField = OutputField(description="Grayscale image of the alpha channel")
    width: int = OutputField(description="The width of the image in pixels")
    height: int = OutputField(description="The height of the image in pixels")


@invocation(
    "rgb_split",
    title="RGB Split",
    tags=["rgb", "image", "color"],
    category="image",
    version="1.0.0",
)
class RGBSplitInvocation(BaseInvocation, WithMetadata, WithBoard):
    """Split an image into RGB color channels and alpha"""

    image: ImageField = InputField(description="The image to split into channels")

    def invoke(self, context: InvocationContext) -> RGBSplitOutput:
        image = context.images.get_pil(self.image.image_name)
        mode = image.mode

        # Ensure we're working with an RGB image
        if mode not in ["RGB", "RGBA"]:
            image = image.convert("RGB")
        
        # Extract the alpha channel if present, otherwise create a solid white one
        alpha_channel = image.getchannel("A") if mode == "RGBA" else Image.new("L", image.size, color=255)
        
        # Split the image into R, G, B channels
        r, g, b = image.split()[:3]  # Only take the first 3 channels (R, G, B)

        # Save each channel as a grayscale image
        image_r_dto = context.images.save(Image.fromarray(numpy.array(r)))
        image_g_dto = context.images.save(Image.fromarray(numpy.array(g)))
        image_b_dto = context.images.save(Image.fromarray(numpy.array(b)))
        image_alpha_dto = context.images.save(Image.fromarray(numpy.array(alpha_channel)))

        return RGBSplitOutput(
            r_channel=ImageField(image_name=image_r_dto.image_name),
            g_channel=ImageField(image_name=image_g_dto.image_name),
            b_channel=ImageField(image_name=image_b_dto.image_name),
            alpha_channel=ImageField(image_name=image_alpha_dto.image_name),
            width=image.width,
            height=image.height,
        )


@invocation(
    "rgb_merge",
    title="RGB Merge",
    tags=["rgb", "image", "color"],
    category="image",
    version="1.0.0",
)
class RGBMergeInvocation(BaseInvocation, WithMetadata, WithBoard):
    """Merge RGB color channels and alpha"""

    r_channel: Optional[ImageField] = InputField(
        default=None,
        description="The red channel",
    )
    g_channel: Optional[ImageField] = InputField(
        default=None,
        description="The green channel",
    )
    b_channel: Optional[ImageField] = InputField(
        default=None,
        description="The blue channel",
    )
    alpha_channel: Optional[ImageField] = InputField(
        default=None,
        description="The alpha channel",
    )

    def invoke(self, context: InvocationContext) -> ImageOutput:
        # Get the images for each channel, defaulting to black if not provided
        r_image, g_image, b_image, alpha_image = None, None, None, None
        
        # Get image size from the first non-None channel
        image_size = None
        for channel in [self.r_channel, self.g_channel, self.b_channel, self.alpha_channel]:
            if channel is not None:
                channel_image = context.images.get_pil(channel.image_name)
                image_size = channel_image.size
                break
                
        if image_size is None:
            raise ValueError("At least one channel must be provided")
            
        # Load each channel or create a black one if missing
        if self.r_channel is not None:
            r_image = context.images.get_pil(self.r_channel.image_name).convert("L")
        else:
            r_image = Image.new("L", image_size, color=0)
            
        if self.g_channel is not None:
            g_image = context.images.get_pil(self.g_channel.image_name).convert("L")
        else:
            g_image = Image.new("L", image_size, color=0)
            
        if self.b_channel is not None:
            b_image = context.images.get_pil(self.b_channel.image_name).convert("L")
        else:
            b_image = Image.new("L", image_size, color=0)

        # Merge the RGB channels
        rgb_image = Image.merge("RGB", (r_image, g_image, b_image))
        
        # Add alpha channel if provided
        if self.alpha_channel is not None:
            alpha_image = context.images.get_pil(self.alpha_channel.image_name).convert("L")
            rgb_image = rgb_image.convert("RGBA")
            rgb_image.putalpha(alpha_image)
            
        # Save and return the merged image
        image_dto = context.images.save(rgb_image)
        return ImageOutput.build(image_dto)

