# Copyright (c) 2023 Darren Ringer <https://github.com/dwringer/>
# Implements the technique described at https://blog.demofox.org/2017/10/25/transmuting-white-noise-to-blue-red-green-purple/

import math
from typing import Literal

import cv2
import numpy
import PIL.Image
import scipy.stats
import torch

from invokeai.app.invocations.baseinvocation import (
    BaseInvocation,
    InvocationContext,
    invocation,
)
from invokeai.app.invocations.fields import ImageField, InputField, WithBoard, WithMetadata
from invokeai.app.invocations.noise import NoiseOutput
from invokeai.app.invocations.primitives import ImageOutput
from invokeai.app.util.misc import SEED_MAX, get_random_seed


def flatten_histogram(image_array, seed=None):
    shape = image_array.shape
    pixel_count = shape[0] * shape[1]
    pixels = image_array.copy().reshape((pixel_count, 1))
    # Track original index of each pixel
    pixels = numpy.concatenate((pixels, numpy.arange(pixel_count).reshape((pixel_count, 1))), axis=1)
    # Shuffle pixels
    prng = numpy.random.default_rng(seed=(get_random_seed() if seed is None else seed))
    prng.shuffle(pixels, axis=0)
    # Reorder pixels
    pixels = pixels[pixels[:, 0].argsort()]
    # Associate current indexes to pixels
    pixels = numpy.concatenate((pixels, numpy.arange(pixel_count).reshape((pixel_count, 1))), axis=1)
    image_array = image_array.reshape((pixel_count, 1))
    if isinstance(image_array.flat[0], numpy.uint8):
        image_array[pixels[:, 1].astype("int"), 0] = (
            255.0 * pixels[:, 2].astype("float") / (pixel_count - 1.0)
        ).astype("uint8")
    else:
        image_array[pixels[:, 1].astype("int"), 0] = pixels[:, 2] / (pixel_count - 1.0)
    image_array = image_array.reshape(shape)
    return image_array


def white_noise_array(width, height, seed=None, is_uint8=True):
    prng = numpy.random.default_rng(get_random_seed() if (seed is None) else seed)
    if is_uint8:
        return (prng.random((height, width, 1)).reshape((height, width)) * 255.0).astype("uint8")
    else:
        return prng.uniform(0.0, 1.0, size=(height, width))


def white_noise_image(width, height, seed=None):
    return PIL.Image.fromarray(
        white_noise_array(width, height, seed=seed),
        mode="L",
    )


def red_noise_array(
    width, height, iterations=5, sigma=0.5, radius=None, blur_threshold=0.005, seed=None, is_uint8=True
):
    # cv default: sigma = 0.3*((ksize-1)*0.5 - 1) + 0.8
    if radius is None:
        #     // returns the number of pixels needed to represent a gaussian kernal that has values
        #     // down to the threshold amount.  A gaussian function technically has values everywhere
        #     // on the image, but the threshold lets us cut it off where the pixels contribute to
        #     // only small amounts that aren't as noticeable.
        #     return int(floor(1.0f + 2.0f * sqrtf(-2.0f * sigma * sigma * log(c_blurThresholdPercent)))) + 1;
        radius = int(math.floor(1.0 + 2.0 * math.sqrt(-2.0 * sigma * sigma * math.log(blur_threshold)))) + 1
    arr = white_noise_array(width, height, seed=seed, is_uint8=is_uint8)
    r = 2 * radius + 1
    s = math.sqrt(sigma * sigma / 2)
    for _i in range(iterations):
        arr = cv2.copyMakeBorder(
            arr,
            radius,
            radius,
            radius,
            radius,
            cv2.BORDER_WRAP,
        )
        arr = cv2.GaussianBlur(arr, (r, r), s, s)
        arr = arr[radius:-radius, radius:-radius].copy()
        arr = flatten_histogram(arr, seed=seed)
    return arr


def red_noise_image(width, height, iterations=5, sigma=0.5, radius=None, blur_threshold=0.005, seed=None):
    return PIL.Image.fromarray(
        red_noise_array(width, height, iterations, sigma, radius, blur_threshold, seed=seed), mode="L"
    )


def blue_noise_array(
    width, height, iterations=5, sigma=1.0, radius=None, blur_threshold=0.005, seed=None, is_uint8=True
):
    if radius is None:
        radius = int(math.floor(1.0 + 2.0 * math.sqrt(-2.0 * sigma * sigma * math.log(blur_threshold)))) + 1
    arr = white_noise_array(width, height, seed=seed, is_uint8=is_uint8)
    r = 2 * radius + 1
    s = math.sqrt(sigma * sigma / 2)
    for _i in range(iterations):
        blurred = cv2.copyMakeBorder(
            arr,
            radius,
            radius,
            radius,
            radius,
            cv2.BORDER_WRAP,
        )
        blurred = cv2.GaussianBlur(blurred, (r, r), s, s)
        arr = arr - blurred[radius:-radius, radius:-radius]
        arr = flatten_histogram(arr, seed=seed)
    return arr


def blue_noise_image(width, height, iterations=5, sigma=0.5, radius=None, blur_threshold=0.005, seed=None):
    return PIL.Image.fromarray(
        blue_noise_array(width, height, iterations, sigma, radius, blur_threshold, seed=seed), mode="L"
    )


def green_noise_array(
    width,
    height,
    iterations=5,
    sigma_strong=2.0,
    sigma_weak=0.5,
    radius_strong=None,
    radius_weak=None,
    blur_threshold=0.005,
    reset_cache=True,
    use_cache=None,
    seed=None,
    is_uint8=True,
):
    if radius_strong is None:
        radius_strong = (
            int(math.floor(1.0 + 2.0 * math.sqrt(-2.0 * sigma_strong * sigma_strong * math.log(blur_threshold)))) + 1
        )
    if radius_weak is None:
        radius_weak = (
            int(math.floor(1.0 + 2.0 * math.sqrt(-2.0 * sigma_weak * sigma_weak * math.log(blur_threshold)))) + 1
        )
    arr = white_noise_array(width, height, seed=seed, is_uint8=is_uint8)
    r_s = 2 * radius_strong + 1
    r_w = 2 * radius_weak + 1
    s_s = math.sqrt(sigma_strong * sigma_strong / 2)
    s_w = math.sqrt(sigma_weak * sigma_weak / 2)
    for _i in range(iterations):
        b_strong = cv2.copyMakeBorder(
            arr,
            radius_strong,
            radius_strong,
            radius_strong,
            radius_strong,
            cv2.BORDER_WRAP,
        )
        b_strong = cv2.GaussianBlur(b_strong, (r_s, r_s), s_s, s_s)
        b_weak = cv2.copyMakeBorder(
            arr,
            radius_weak,
            radius_weak,
            radius_weak,
            radius_weak,
            cv2.BORDER_WRAP,
        )
        b_weak = cv2.GaussianBlur(b_weak, (r_w, r_w), s_w, s_w)
        arr = (
            b_weak[radius_weak:-radius_weak, radius_weak:-radius_weak].copy()
            - b_strong[radius_strong:-radius_strong, radius_strong:-radius_strong]
        )
        arr = flatten_histogram(arr, seed=seed)
    return arr


def green_noise_image(
    width,
    height,
    iterations=5,
    sigma_strong=2.0,
    sigma_weak=0.5,
    radius_strong=None,
    radius_weak=None,
    blur_threshold=0.005,
    seed=None,
):
    return PIL.Image.fromarray(
        (
            green_noise_array(
                width,
                height,
                iterations,
                sigma_strong,
                sigma_weak,
                radius_strong,
                radius_weak,
                blur_threshold,
                seed=seed,
                is_uint8=False,
            )
            * 255.0
        ).astype("uint8"),
        mode="L",
    )


@invocation(
    "noiseimg_2d",
    title="2D Noise Image",
    tags=["image", "noise"],
    category="image",
    version="1.2.0",
)
class NoiseImage2DInvocation(BaseInvocation, WithMetadata, WithBoard):
    """Creates an image of 2D Noise approximating the desired characteristics"""

    noise_type: Literal["White", "Red", "Blue", "Green"] = InputField(  # TODO: Pyramid Noise
        default="White", description="Desired noise spectral characteristics"
    )
    width: int = InputField(default=512, description="Desired image width")
    height: int = InputField(default=512, description="Desired image height")
    seed: int = InputField(default=0, ge=0, le=SEED_MAX, description="Seed for noise generation")
    iterations: int = InputField(default=15, description="Noise approx. iterations")
    blur_threshold: float = InputField(
        default=0.2, description="Threshold used in computing noise (lower is better/slower)"
    )
    sigma_red: float = InputField(default=3.0, description="Sigma for strong gaussian blur LPF for red/green")
    sigma_blue: float = InputField(default=1.0, description="Sigma for weak gaussian blur HPF for blue/green")

    def invoke(self, context: InvocationContext) -> ImageOutput:
        image = None
        if self.noise_type == "White":
            image = white_noise_image(self.width, self.height, seed=self.seed)
        elif self.noise_type == "Red":
            image = red_noise_image(
                self.width,
                self.height,
                iterations=self.iterations,
                sigma=self.sigma_red,
                blur_threshold=self.blur_threshold / 100.0,
                seed=self.seed,
            )
        elif self.noise_type == "Blue":
            image = blue_noise_image(
                self.width,
                self.height,
                iterations=self.iterations,
                sigma=self.sigma_blue,
                blur_threshold=self.blur_threshold / 100.0,
                seed=self.seed,
            )
        elif self.noise_type == "Green":
            image = green_noise_image(
                self.width,
                self.height,
                iterations=self.iterations,
                sigma_strong=self.sigma_red,
                sigma_weak=self.sigma_blue,
                blur_threshold=self.blur_threshold / 100.0,
                seed=self.seed,
            )

        image_dto = context.images.save(image)

        return ImageOutput.build(image_dto)


@invocation(
    "noise_spectral",
    title="Noise (Spectral characteristics)",
    tags=["noise"],
    category="noise",
    version="1.2.0",
)
class NoiseSpectralInvocation(BaseInvocation):
    """Creates an image of 2D Noise approximating the desired characteristics"""

    noise_type: Literal["White", "Red", "Blue", "Green"] = InputField(
        default="White", description="Desired noise spectral characteristics"
    )
    width: int = InputField(default=512, description="Desired image width")
    height: int = InputField(default=512, description="Desired image height")
    seed: int = InputField(default=0, ge=0, le=SEED_MAX, description="Seed for noise generation")
    iterations: int = InputField(default=15, description="Noise approx. iterations")
    blur_threshold: float = InputField(
        default=0.2, description="Threshold used in computing noise (lower is better/slower)"
    )
    sigma_red: float = InputField(default=3.0, description="Sigma for strong gaussian blur LPF for red/green")
    sigma_blue: float = InputField(default=1.0, description="Sigma for weak gaussian blur HPF for blue/green")

    def invoke(self, context: InvocationContext) -> NoiseOutput:
        latents = None
        w, h = self.width // 8, self.height // 8

        def torchify(arr):
            epsilon = numpy.finfo(float).eps
            arr = numpy.where(numpy.equal(arr, 0.0), epsilon, arr)
            arr = (scipy.stats.boxcox(arr.flatten())[0]).reshape(arr.shape)
            arr -= arr.mean()
            arr /= math.sqrt(arr.var())
            arr = torch.from_numpy(arr)
            return arr

        if self.noise_type == "White":
            latents = (
                torch.stack([torchify(white_noise_array(w, h, seed=self.seed + i, is_uint8=False)) for i in range(4)])
                .unsqueeze(0)
                .to("cpu")
            )
        elif self.noise_type == "Red":
            latents = (
                torch.stack(
                    [
                        torchify(
                            red_noise_array(
                                w,
                                h,
                                iterations=self.iterations,
                                sigma=self.sigma_red,
                                blur_threshold=self.blur_threshold / 100.0,
                                seed=self.seed + i,
                                is_uint8=False,
                            )
                        )
                        for i in range(4)
                    ]
                )
                .unsqueeze(0)
                .to("cpu")
            )
        elif self.noise_type == "Blue":
            latents = (
                torch.stack(
                    [
                        torchify(
                            blue_noise_array(
                                w,
                                h,
                                iterations=self.iterations,
                                sigma=self.sigma_blue,
                                blur_threshold=self.blur_threshold / 100.0,
                                seed=self.seed + i,
                                is_uint8=False,
                            )
                        )
                        for i in range(4)
                    ]
                )
                .unsqueeze(0)
                .to("cpu")
            )
        elif self.noise_type == "Green":
            latents = (
                torch.stack(
                    [
                        torchify(
                            green_noise_array(
                                w,
                                h,
                                iterations=self.iterations,
                                sigma_strong=self.sigma_red,
                                sigma_weak=self.sigma_blue,
                                blur_threshold=self.blur_threshold / 100.0,
                                seed=self.seed + i,
                                is_uint8=False,
                            )
                        )
                        for i in range(4)
                    ]
                )
                .unsqueeze(0)
                .to("cpu")
            )

        name = context.tensors.save(tensor=latents)
        return NoiseOutput.build(latents_name=name, latents=latents, seed=self.seed)


@invocation(
    "flatten_histogram_mono",
    title="Flatten Histogram (Grayscale)",
    tags=["noise", "image"],
    category="image",
    version="1.2.0",
)
class FlattenHistogramMono(BaseInvocation, WithMetadata, WithBoard):
    """Scales the values of an L-mode image by scaling them to the full range 0..255 in equal proportions"""

    image: ImageField = InputField(description="Single-channel image for which to flatten the histogram")

    def invoke(self, context: InvocationContext) -> ImageOutput:
        image = context.images.get_pil(self.image.image_name)
        if not (image.mode == "L"):
            image = image.convert("L")
        image = PIL.Image.fromarray(flatten_histogram(numpy.array(image)), mode="L")
        image_dto = context.images.save(image)

        return ImageOutput.build(image_dto)
