from typing import Optional

import torch
from PIL import Image
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


def tensor_from_pil_image(img, normalize=False):
    return image_resized_to_grid_as_tensor(img, normalize=normalize, multiple_of=1)


@invocation(
    'latent_som',
    title="Latent Quantize (SOM)",
    tags=["latents", "quantize", "som"],
    category="latents",
    version="0.0.1"
)
class LatentSOMInvocation(BaseInvocation):
    """Use a self-organizing map to quantize the values of a latent tensor"""
    latents_in: LatentsField = InputField(description="The latents tensor to quantize")
    reference_in: Optional[LatentsField] = InputField(
        default=None,
        description="Optional alternate latents to use for training"
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
                

        som_tensor = torch.zeros((1, 4, self.height, self.width))

        def bmu(latent_pixel):

            diffs = torch.add(torch.mul(latent_pixel, -1.0), som_tensor)

            sums_of_squares = torch.sum(torch.square(diffs), 1)  # 1, 1, h, w

            bmu_value, bmu_indices = torch.min(sums_of_squares, 1)
            bmu_min, bmu_min_index = torch.min(bmu_value, 1)

            return int(bmu_indices[0, bmu_min_index]), int(bmu_min_index)


        sample_indices = torch.randperm(latents_in.shape[2] * latents_in.shape[3])
        sample = latents_in.view(1, 4, -1)[:, :, sample_indices]

        init_sample = sample[:, :, :(self.width * self.height)].view(1, 4, self.height, self.width)

        som_tensor = torch.clone(init_sample)
        row_indices = torch.arange(
            0,
            som_tensor.shape[2]
        ).expand(som_tensor.shape[3], -1).transpose(0, 1)
        column_indices = torch.arange(0, som_tensor.shape[3]).expand(som_tensor.shape[2], -1)
        
        # LATENTS_IN.SHAPE/SIZE() = 1, 4, H, W)

        neighborhood_width_max = som_tensor.shape[2] + som_tensor.shape[3] - 2
        
        for i in range(self.steps):
            sample_indices = torch.randperm(latents_in.shape[2] * latents_in.shape[3])
            sample = latents_in.view(1, 4, -1)[:, :, sample_indices].view(latents_in.shape)

            neighborhood_width = neighborhood_width_max - (
                neighborhood_width_max * float(i) / self.steps
            )
            
            for j in range(latents_in.shape[2]):
                for k in range(latents_in.shape[3]):
                    latent_pixel = sample[:, :, j, k].expand(
                        self.height,
                        self.width,
                        1,
                        4
                    ).movedim(0, 3).movedim(0, 3)  # h, w, 1, 4 -> 1, 4, h, w
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
                    4
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

   
@invocation(
    'image_som',
    title="Image Quantize (Kohonen map)",
    tags=["image", "color", "quantize", "som", "kohonen", "palette"],
    category="image",
    version="0.6.2"
)
class ImageSOMInvocation(BaseInvocation):
    """Use a Kohonen self-organizing map to quantize the pixel values of an image"""
    image_in: ImageField = InputField(description="The image to quantize")
    map_in: Optional[ImageField] = InputField(
        default=None,
        description="Use an existing SOM instead of training one (skips all training)"
    )
    map_width:  int = InputField(default=16,   description="Width (in cells) of the self-organizing map to train")
    map_height: int = InputField(default=16,   description="Height (in cells) of the self-organizing map to train")
    steps:  int = InputField(default=128, description="Training step count for the self-organizing map")
    training_scale: float = InputField(default=0.25, description="Nearest-neighbor scale image size prior to sampling")
    sample_width: Optional[int] = InputField(default=64, description="Width of randomly selected sample per step")
    sample_height: Optional[int] = InputField(default=64, description="Height of randomly selected sample per step")


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
            image_in = image_in.expand(*([1] + list(image_in.shape)))

        image_out = torch.clone(image_in)

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
            bmus_i = torch.empty((image_out.shape[2], image_out.shape[3]), dtype=bmus_j.dtype)

            for i in range(image_in.shape[2]):
                row_bmus_j = som_idx_j[0, i]
                bmus_i[i] = som_idx_i[0, i, torch.arange(som_idx_i.shape[2]), row_bmus_j]

            image_out = som_tensor[:,:,bmus_i,bmus_j]
        else:
            bmus_i = torch.empty((image_out.shape[2], image_out.shape[3]), dtype=torch.int32)
            bmus_j = torch.empty((image_out.shape[2], image_out.shape[3]), dtype=torch.int32)
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

            image_out = som_tensor[:,:,bmus_i,bmus_j]

        image_out = pil_image_from_tensor(image_out[0], mode="RGB")
        image_dto = context.images.save(image_out)

        map_out = pil_image_from_tensor(som_tensor[0], mode="RGB")
        som_dto = context.images.save(map_out)

        return ImageSOMOutput(
            image_out=ImageField(image_name=image_dto.image_name),
            map_out=ImageField(image_name=som_dto.image_name),
            image_width=image_dto.width,
            image_height=image_dto.height,
            map_width=som_dto.width,
            map_height=som_dto.height
        )
