from typing import Literal, Optional

import numpy as np
import torch
from PIL import Image
from scipy.optimize import linear_sum_assignment
from torchvision.transforms.functional import to_pil_image as pil_image_from_tensor

from invokeai.backend.stable_diffusion.diffusers_pipeline import image_resized_to_grid_as_tensor
from invokeai.invocation_api import (
    BaseInvocation,
    BaseInvocationOutput,
    ImageField,
    ImageOutput,
    InputField,
    InvocationContext,
    LatentsField,
    LatentsOutput,
    OutputField,
    WithBoard,
    WithMetadata,
    invocation,
    invocation_output,
)

from invokeai.backend.image_util.composition import (
    linear_srgb_from_oklab,
    linear_srgb_from_srgb,
    oklab_from_linear_srgb,
    srgb_from_linear_srgb,
 )


def tensor_from_pil_image(img, normalize=False):
    return image_resized_to_grid_as_tensor(img, normalize=normalize, multiple_of=1)


def calculate_corner_distances(corners1, corners2):
    """
    Calculate Euclidean distances between corresponding corners' [Ok]LAB color coordinates.

    Args:
        corners1 (list): RGB tuples of corners from first image
        corners2 (list): RGB tuples of corners from second image

    Returns:
        float: Total Euclidean distance between corners
    """

    corners1 = oklab_from_linear_srgb(linear_srgb_from_srgb(torch.tensor(corners1).unsqueeze(0))).squeeze(0)
    corners2 = oklab_from_linear_srgb(linear_srgb_from_srgb(torch.tensor(corners2).unsqueeze(0))).squeeze(0)

    return sum(
        np.linalg.norm(np.array(c1) - np.array(c2)) 
        for c1, c2 in zip(corners1, corners2)
    )


def get_image_corners(img):
    """
    Extract corner pixel RGB values from an image.

    Args:
        img (PIL.Image): Input image

    Returns:
        list: RGB tuples of corners [top-left, top-right, bottom-left, bottom-right]
    """
    width, height = img.size
    corners = [
        img.getpixel((0, 0)),           # Top-left
        img.getpixel((width-1, 0)),     # Top-right
        img.getpixel((0, height-1)),    # Bottom-left
        img.getpixel((width-1, height-1)) # Bottom-right
    ]
    return corners


def find_best_transformation(img1, img2):
    """
    Find the best image rotation/flip transformation to minimize corner distances.

    Args:
        img1 (PIL.Image): Reference image
        img2 (PIL.Image): Image to transform

    Returns:
        tuple: (best_rotation, best_flip, min_distance)
    """
    corners1 = get_image_corners(img1)

    # Possible transformations: rotations and flips
    rotations = [0, 90, 180, 270]
    flip_modes = [None, 'horizontal', 'vertical']

    best_rotation = 0
    best_flip = None
    min_distance = float('inf')

    for rotation in rotations:
        for flip in flip_modes:
            # Create transformed image
            transformed_img = img2.copy()

            # Apply rotation
            if 0 < rotation:
                transformed_img = transformed_img.rotate(rotation)

            # Apply flip
            if flip == 'horizontal':
                transformed_img = transformed_img.transpose(Image.FLIP_LEFT_RIGHT)
            elif flip == 'vertical':
                transformed_img = transformed_img.transpose(Image.FLIP_TOP_BOTTOM)

            # Get corners of transformed image
            corners2 = get_image_corners(transformed_img)

            # Calculate total corner distance
            distance = calculate_corner_distances(corners1, corners2)

            # Update best transformation if distance is smaller
            if distance < min_distance:
                min_distance = distance
                best_rotation = rotation
                best_flip = flip

    return best_rotation, best_flip, min_distance


def apply_transformation(img2, rotation, flip):
    """
    Apply a preset transformation to an image.

    Args:
        img2 (PIL.Image): Image to transform
        rotation (int): Rotation angle
        flip (str): Flip type

    Returns:
        PIL.Image: Transformed image
    """
    transformed_img = img2.copy()

    # Apply rotation
    if rotation:
        transformed_img = transformed_img.rotate(rotation)

    # Apply flip
    if flip == 'horizontal':
        transformed_img = transformed_img.transpose(Image.FLIP_LEFT_RIGHT)
    elif flip == 'vertical':
        transformed_img = transformed_img.transpose(Image.FLIP_TOP_BOTTOM)

    return transformed_img


@invocation(
    'latent_som',
    title="Latent Quantize (Kohonen map)",
    tags=["latents", "quantize", "som", "kohonen"],
    category="latents",
    version="0.0.3"
)
class LatentSOMInvocation(BaseInvocation):
    """Use a self-organizing map to quantize the values of a latent tensor.

This is highly experimental and not really suitable for most use cases. It's very easy to use settings that will appear to hang, or tie up the PC for a very long time, so use of this node is somewhat discouraged.
"""
    latents_in: LatentsField = InputField(description="The latents tensor to quantize")
    reference_in: Optional[LatentsField] = InputField(
        default=None,
        description="Optional alternate latents to use for training",
    )
    width:  int = InputField(default=4,   description="Width (in cells) of the self-organizing map")
    height: int = InputField(default=3,   description="Height (in cells) of the self-organizing map")
    steps:  int = InputField(default=256, description="Training step count for the self-organizing map")

    def invoke(self, context: InvocationContext) -> LatentsOutput:
        latents_in = None
        if self.reference_in is not None:
            latents_in = context.tensors.load(self.reference_in.latents_name)
        else:
            latents_in = context.tensors.load(self.latents_in.latents_name)

        num_channels = latents_in.shape[1]

        som_tensor = torch.zeros((1, num_channels, self.height, self.width))

        def bmu(latent_pixel):

            diffs = torch.add(torch.mul(latent_pixel, -1.0), som_tensor)

            sums_of_squares = torch.sum(torch.square(diffs), 1)  # 1, 1, h, w

            bmu_value, bmu_indices = torch.min(sums_of_squares, 1)
            bmu_min, bmu_min_index = torch.min(bmu_value, 1)

            return int(bmu_indices[0, bmu_min_index]), int(bmu_min_index)


        sample_indices = torch.randperm(latents_in.shape[2] * latents_in.shape[3])
        sample = latents_in.view(1, num_channels, -1)[:, :, sample_indices]

        init_sample = sample[:, :, :(self.width * self.height)].view(1, num_channels, self.height, self.width)

        som_tensor = torch.clone(init_sample)
        row_indices = torch.arange(
            0,
            som_tensor.shape[2]
        ).expand(som_tensor.shape[3], -1).transpose(0, 1)
        column_indices = torch.arange(0, som_tensor.shape[3]).expand(som_tensor.shape[2], -1)
        
        # LATENTS_IN.SHAPE/SIZE() = 1, num_channels, H, W)

        neighborhood_width_max = som_tensor.shape[2] + som_tensor.shape[3] - 2
        
        for i in range(self.steps):
            sample_indices = torch.randperm(latents_in.shape[2] * latents_in.shape[3])
            sample = latents_in.view(1, num_channels, -1)[:, :, sample_indices].view(latents_in.shape)

            neighborhood_width = neighborhood_width_max - (
                neighborhood_width_max * float(i) / self.steps
            )
            
            for j in range(latents_in.shape[2]):
                for k in range(latents_in.shape[3]):
                    latent_pixel = sample[:, :, j, k].expand(
                        self.height,
                        self.width,
                        1,
                        num_channels
                    ).movedim(0, 3).movedim(0, 3)  # h, w, 1, num_channels -> 1, 4, h, num_channels
                    bmu_row, bmu_column = bmu(latent_pixel)
                    grid_distances = torch.add(
                        torch.abs(torch.sub(row_indices, bmu_row)),
                        torch.abs(torch.sub(column_indices, bmu_column))
                    )
                    theta = torch.exp(
                        torch.div(
                            torch.mul(torch.square(grid_distances), -1),
                            2 * neighborhood_width ** 2
                        )
                    )

                    som_tensor = torch.add(
                        som_tensor,
                        torch.mul(
                            theta,
                            torch.mul(
                                torch.add(
                                    torch.mul(
                                        som_tensor,
                                        -1
                                    ),
                                    latent_pixel
                                ),
                                1 - (i / self.steps)  # learning rate schedule
                            )
                        )
                    )
                    
        if self.reference_in is not None:
            latents_in = context.tensors.load(self.latents_in.latents_name)
        latents_out = torch.clone(latents_in)
        for i in range(latents_in.shape[2]):
            for j in range(latents_in.shape[3]):
                latent_pixel = latents_in[:, :, i, j].expand(
                    self.height,
                    self.width,
                    1,
                    num_channels
                ).movedim(0, 3).movedim(0, 3)
                bmu_i, bmu_j = bmu(latent_pixel)
                latents_out[:, :, i, j] = som_tensor[:, :, bmu_i, bmu_j]
        return LatentsOutput.build(
            latents_name=context.tensors.save(tensor=latents_out),
            latents=latents_out,
            seed=self.latents_in.seed
        )


@invocation_output('image_som_output')
class ImageSOMOutput(BaseInvocationOutput):
    """Outputs an image and the SOM used to quantize that image"""
    image_out: ImageField = OutputField(description="Quantized image")
    map_out: ImageField = OutputField(description="The pixels of the self-organizing map")
    image_width: int = OutputField(description="Width of the quantized image")
    image_height: int = OutputField(description="Height of the quantized image")
    map_width: int = OutputField(description="Width of the SOM image")
    map_height: int = OutputField(description="Height of the SOM image")


SWAP_MODES = [
    'Direct',
    'Reorient corners',
    'Minimize distances',
]
   
@invocation(
    'image_som',
    title="Image Quantize (Kohonen map)",
    tags=["image", "color", "quantize", "som", "kohonen", "palette"],
    category="image",
    version="0.8.2"
)
class ImageSOMInvocation(BaseInvocation):
    """Use a Kohonen self-organizing map to quantize the pixel values of an image.

It's possible to pass in an existing map, which will be used instead of training a new one. It's also possible to pass in a "swap map", which will be used in place of the standard map's assigned pixel values in quantizing the target image - these values can be correlated either one by one by a linear assignment minimizing the distances\* between each of them, or by swapping their coordinates on the maps themselves, which can be oriented first such that their corner distances\* are minimized achieving a closest-fit while attempting to preserve mappings of adjacent colors.

\*Here, "distances" refers to the euclidean distances between (L, a, b) tuples in Oklab color space.
"""

    image_in: ImageField = InputField(description="The image to quantize")
    map_in: Optional[ImageField] = InputField(
        default=None,
        description="Use an existing SOM instead of training one (skips all training)",
    )
    swap_map: Optional[ImageField] = InputField(
        default=None,
        description="Take another map and swap in its colors after obtaining best-match indices but prior to mapping",
    )
    swap_mode: Literal[tuple(SWAP_MODES)] = InputField(default=SWAP_MODES[2], description="How to employ the swap map - directly, reoriented or rearranged")
    map_width: int = InputField(default=16, description="Width (in cells) of the self-organizing map to train")
    map_height: int = InputField(default=16, description="Height (in cells) of the self-organizing map to train")
    steps: int = InputField(default=64, description="Training step count for the self-organizing map")
    training_scale: float = InputField(
        default=0.25,
        description="Nearest-neighbor scale image size prior to sampling - size close to sample size is recommended"
    )
    sample_width:   int   = InputField(
        default=64,
        description="Width of assorted pixel sample per step - for performance, keep this number low"
    )
    sample_height:  int   = InputField(
        default=64,
        description="Height of assorted pixel sample per step - for performance, keep this number low"
    )


    def invoke(self, context: InvocationContext) -> ImageSOMOutput:
        image_in = None

        if self.map_in is not None:
            image_in = context.images.get_pil(self.map_in.image_name)
        else:
            image_in = context.images.get_pil(self.image_in.image_name)

            if not (self.training_scale == 1.0):
                image_in = image_in.resize(
                    (int(image_in.width*self.training_scale), int(image_in.height*self.training_scale)),
                    Image.NEAREST
                )

            if (image_in.width * image_in.height) < (self.sample_width * self.sample_height):
                image_in = image_in.resize(
                    (self.sample_width, self.sample_height),
                    Image.NEAREST
                )

        image_in = image_in.convert('RGB')
        image_in = tensor_from_pil_image(image_in, normalize=False)
        image_in = oklab_from_linear_srgb(linear_srgb_from_srgb(image_in))
        image_in = image_in.expand(*([1] + list(image_in.shape)))

        som_tensor = None
        if self.map_in is not None:
            som_tensor = image_in

        if self.map_in is None:
            som_tensor = torch.zeros((1, 3, self.map_height, self.map_width))

            def bmu(image_pixel):

                diffs = torch.add(torch.mul(image_pixel, -1.0), som_tensor)

                sums_of_squares = torch.sum(torch.square(diffs), 1)  # 1, 3->1, h, w

                bmu_value, bmu_indices = torch.min(sums_of_squares, 1)
                bmu_min, bmu_min_index = torch.min(bmu_value, 1)

                return int(bmu_indices[0, bmu_min_index]), int(bmu_min_index)


            sample_indices = torch.randperm(image_in.shape[2] * image_in.shape[3])
            sample = image_in.view(1, 3, -1)[:, :, sample_indices]

            init_sample = sample[:, :, :(self.map_width * self.map_height)].view(
                1, 3, self.map_height, self.map_width
            )

            som_tensor = torch.clone(init_sample)
            row_indices = torch.arange(
                0,
                som_tensor.shape[2]
            ).expand(som_tensor.shape[3], -1).transpose(0, 1)
            column_indices = torch.arange(0, som_tensor.shape[3]).expand(som_tensor.shape[2], -1)

            # IMAGE_IN.SHAPE/SIZE() = 1, 3, H, W)

            neighborhood_width_max = som_tensor.shape[2] + som_tensor.shape[3] - 2

            for i in range(self.steps):
                sample_indices = torch.randperm(image_in.shape[2] * image_in.shape[3])
                sample = image_in.view(1, 3, -1)[:, :, sample_indices]
                sample = sample[:, :, :int(self.sample_height*self.sample_width)].view(
                    1, 3, self.sample_height, self.sample_width
                )

                neighborhood_width = neighborhood_width_max - (
                    neighborhood_width_max * float(i) / self.steps
                )

                for j in range(sample.shape[2]):
                    for k in range(sample.shape[3]):
                        image_pixel = sample[:, :, j, k].expand(
                            self.map_height,
                            self.map_width,
                            1,
                            3
                        ).movedim(0, 3).movedim(0, 3)  # h, w, 1, 3 -> 1, 3, h, w
                        bmu_row, bmu_column = bmu(image_pixel)
                        grid_distances = torch.add(
                            torch.abs(torch.sub(row_indices, bmu_row)),
                            torch.abs(torch.sub(column_indices, bmu_column))
                        )
                        theta = torch.exp(
                            torch.div(
                                torch.mul(torch.square(grid_distances), -1),
                                2 * neighborhood_width ** 2
                            )
                        )

                        som_tensor = torch.add(
                            som_tensor,
                            torch.mul(
                                theta,
                                torch.mul(
                                    torch.add(
                                        torch.mul(
                                            som_tensor,
                                            -1
                                        ),
                                        image_pixel
                                    ),
                                    1 - (i / self.steps)  # learning rate schedule
                                )
                            )
                        )
                    
        if (self.map_in is not None) or (not (self.training_scale == 1.0)):
            image_in = context.images.get_pil(self.image_in.image_name)
            image_in = image_in.convert('RGB')
            image_in = tensor_from_pil_image(image_in, normalize=False)
            image_in = oklab_from_linear_srgb(linear_srgb_from_srgb(image_in))
            image_in = image_in.expand(*([1] + list(image_in.shape)))

        image_out = torch.clone(image_in)

        bmus_i = torch.empty((image_out.shape[2], image_out.shape[3]), dtype=torch.int32)
        bmus_j = torch.empty((image_out.shape[2], image_out.shape[3]), dtype=torch.int32)
        
        if (  # Do the entire computation at once if the map is small enough
                (som_tensor.shape[2] * som_tensor.shape[3] * image_out.shape[2] * image_out.shape[3]) <=
                (32 * 32 * 1920 * 1080)  # TODO: Need to determine limits...
        ):
            # For every pair of som coordinates, this contains the entire image:
            image_pixels = image_out.expand(
                *(list(som_tensor.shape[-2:])+list(image_out.shape))
            ).movedim(0,5).movedim(0,5)
            # For every pair of image coordinates, this contains the entire SOM:
            som_pixels = som_tensor.expand(
                *(list(image_out.shape[-2:])+list(som_tensor.shape))
            ).movedim(0,3).movedim(0,3)
            # 1, 3, img_h, img_w, som_h, som_w

            # best matching units by least sums of squares of elementwise distances
            sos = torch.sum(torch.square(torch.sub(som_pixels, image_pixels)), 1)
            som_min_i, som_idx_i = torch.min(sos, 3)
            som_min_j, som_idx_j = torch.min(som_min_i, 3)
            # _min is (1, h, w) min distances, while som_idx_j is (1, h, w) bmu indices

            bmus_j = som_idx_j[0]

            for i in range(image_in.shape[2]):
                row_bmus_j = som_idx_j[0, i]
                bmus_i[i] = som_idx_i[0, i, torch.arange(som_idx_i.shape[2]), row_bmus_j]

        else:
            for i in range(image_out.shape[2]):

                # For every pair of som coordinates, this contains the entire image row:
                row_img_pixels = image_out[:,:,i,:].expand(
                    *(list(som_tensor.shape[-2:])+list(image_out.shape[:2])+[image_out.shape[3]])
                ).movedim(0,4).movedim(0,4)
                # For every column coordinate of the image row, this contains the entire SOM:
                row_som_pixels = som_tensor.expand(
                    *([image_out.shape[3]]+list(som_tensor.shape))
                ).movedim(0,2)

                # best matching units by least sums of squares of elementwise distances
                row_sos = torch.sum(torch.square(torch.sub(row_som_pixels, row_img_pixels)), 1)
                row_som_min_i, row_som_idx_i = torch.min(row_sos, 2)
                row_som_min_j, row_som_idx_j = torch.min(row_som_min_i, 2)

                bmus_j[i] = row_som_idx_j[0]

                bmus_i[i] = row_som_idx_i[0, torch.arange(row_som_idx_i.shape[1]), bmus_j[i]]

        if self.swap_map is not None:
            swap_map = context.images.get_pil(self.swap_map.image_name)
            swap_map = swap_map.convert('RGB')
            swap_map = tensor_from_pil_image(swap_map, normalize=False)
            swap_map = oklab_from_linear_srgb(linear_srgb_from_srgb(swap_map))
            swap_map = swap_map.expand(*([1] + list(swap_map.shape)))

            n_elements = som_tensor.shape[2] * som_tensor.shape[3]
            expanded_som = torch.clone(
                som_tensor.view((3, n_elements))
            ).expand((n_elements, 3, n_elements)).movedim(0, 2)
            expanded_swap_map = torch.clone(
                swap_map.view((3, n_elements))
            ).expand((n_elements, 3, n_elements)).movedim(0, 1)

            if self.swap_mode == 'Minimize distances':
                swap_buffer = torch.clone(swap_map)
                cost_matrix = torch.sum(torch.square(torch.sub(expanded_som, expanded_swap_map)), 0).detach().cpu().numpy()
                i_indices, j_indices = linear_sum_assignment(cost_matrix)
                for i, j in zip(i_indices, j_indices):
                    swap_map.view((3, n_elements))[:, i] = swap_buffer.view((3, n_elements))[:,j]
            elif self.swap_mode == 'Reorient corners':
                swap_map_img = context.images.get_pil(self.swap_map.image_name)
                som_tensor_img = pil_image_from_tensor(srgb_from_linear_srgb(linear_srgb_from_oklab(som_tensor.squeeze(0))), mode="RGB")
                
                rotation, flip, min_distance = find_best_transformation(som_tensor_img, swap_map_img)
                transformed_img = apply_transformation(swap_map_img, rotation, flip)
                swap_map = tensor_from_pil_image(transformed_img, normalize=False)
                swap_map = oklab_from_linear_srgb(linear_srgb_from_srgb(swap_map))
                swap_map = swap_map.expand(*([1] + list(swap_map.shape)))
            
            som_tensor = swap_map  # [:,:,bmus_i, bmus_j]

        image_out = som_tensor[:,:,bmus_i,bmus_j]
        image_out = srgb_from_linear_srgb(linear_srgb_from_oklab(image_out[0]))
        image_out = pil_image_from_tensor(image_out, mode="RGB")
        image_dto = context.images.save(image_out)

        som_tensor = srgb_from_linear_srgb(linear_srgb_from_oklab(som_tensor[0]))
        map_out = pil_image_from_tensor(som_tensor, mode="RGB")
        som_dto = context.images.save(map_out)

        return ImageSOMOutput(
            image_out=ImageField(image_name=image_dto.image_name),
            map_out=ImageField(image_name=som_dto.image_name),
            image_width=image_dto.width,
            image_height=image_dto.height,
            map_width=som_dto.width,
            map_height=som_dto.height
        )






