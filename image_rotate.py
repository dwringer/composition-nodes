from math import cos, sin, sqrt
from math import pi as PI

import cv2
import numpy
import PIL.Image

from invokeai.app.invocations.baseinvocation import (
    BaseInvocation,
    InputField,
    InvocationContext,
    WithMetadata,
    invocation,
)
from invokeai.app.invocations.primitives import ImageField, ImageOutput
from invokeai.app.services.image_records.image_records_common import ImageCategory, ResourceOrigin
from invokeai.backend.stable_diffusion.diffusers_pipeline import (
    image_resized_to_grid_as_tensor,
)


def tensor_from_pil_image(img, normalize=False):
    return image_resized_to_grid_as_tensor(
        img, normalize=normalize, multiple_of=1
    )


@invocation(
    "rotate_image",
    title="Rotate/Flip Image",
    tags=["image", "rotate", "flip"],
    category="image",
    version="1.1.0",
)
class ImageRotateInvocation(BaseInvocation, WithMetadata):
    """Rotates an image by a given angle (in degrees clockwise)."""
    image: ImageField = InputField(
        default=None, description="Image to be rotated clockwise"
    )
    degrees: float = InputField(
        default=90., description="Angle (in degrees clockwise) by which to rotate"
    )
    expand_to_fit: bool = InputField(
        default=True, description="If true, extends the image boundary to fit the rotated content"
    )
    flip_horizontal: bool = InputField(
        default=False, description="If true, flips the image horizontally"
    )
    flip_vertical: bool = InputField(
        default=False, description="If true, flips the image vertically"
    )

    def invoke(self, context: InvocationContext) -> ImageOutput:
        image_in = context.services.images.get_pil_image(self.image.image_name)

        # TODO: Preserve mode, alpha
        image_in = image_in.convert('RGB')

        image_width, image_height = image_in.size

        center_x = image_width // 2
        center_y = image_height // 2

        rotation_matrix = cv2.getRotationMatrix2D(
            (center_x, center_y),
            -1.*self.degrees,
            1.
        )
        translation_matrix = None

        radians = PI * self.degrees / 180.
        diagonal = sqrt(image_width**2.+image_height**2.)
        theta_diagonal = numpy.arctan2(image_height, image_width)

        new_height, new_width = image_height, image_width
        if self.expand_to_fit:
            new_height = int(
                diagonal * max(abs(sin(radians + theta_diagonal)),
                               abs(sin(radians - theta_diagonal)))
            )
            new_width = int(
                diagonal * max(abs(cos(radians + theta_diagonal)),
                               abs(cos(radians - theta_diagonal)))
            )
            translation_matrix = numpy.float64(
                [
                    [1, 0, (new_width - image_width) // 2],
                    [0, 1, (new_height - image_height) // 2]
                ]
            )

        transformation = rotation_matrix
        if translation_matrix is not None:
            transformation = (
                numpy.vstack((translation_matrix, numpy.array([0, 0, 1]))) @  \
                numpy.vstack((transformation, numpy.array([0, 0, 1])))
            )[:2,:]

        rgb_nparray = tensor_from_pil_image(image_in).movedim(0, 2).numpy()

        rgb_nparray = (cv2.warpAffine(
            rgb_nparray, transformation, (new_width, new_height)
        ) * 255.).astype('uint8')

        if self.flip_vertical:
            rgb_nparray = numpy.flip(rgb_nparray, axis=0)
        if self.flip_horizontal:
            rgb_nparray = numpy.flip(rgb_nparray, axis=1)

        image_out = PIL.Image.fromarray(rgb_nparray, mode="RGB")

        image_dto = context.services.images.create(
            image=image_out,
            image_origin=ResourceOrigin.INTERNAL,
            image_category=ImageCategory.GENERAL,
            node_id=self.id,
            session_id=context.graph_execution_state_id,
            is_intermediate=self.is_intermediate,
            metadata=self.metadata,
            workflow=context.workflow,
        )

        return ImageOutput(
            image=ImageField(image_name=image_dto.image_name),
            width=image_dto.width,
            height=image_dto.height,
        )
