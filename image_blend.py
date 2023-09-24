# TODO: Improve blend modes
# TODO: Add nodes like Hue Adjust for Saturation/Contrast/etc... ?
# TODO: Continue implementing more blend modes/color spaces(?)
# TODO: Custom ICC profiles with PIL.ImageCms?
# TODO: Blend multiple layers all crammed into a tensor(?) or list

# Copyright (c) 2023 Darren Ringer <dwringer@gmail.com>
# Parts based on Oklab: Copyright (c) 2021 Björn Ottosson <https://bottosson.github.io/>
# HSL code based on CPython: Copyright (c) 2001-2023 Python Software Foundation; All Rights Reserved
from io import BytesIO
import os.path
from math import pi as PI
from typing import Literal, Optional

import PIL.Image, PIL.ImageOps, PIL.ImageCms
import torch
from torchvision.transforms.functional import to_pil_image as pil_image_from_tensor

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


MAX_FLOAT = torch.finfo(torch.tensor(1.).dtype).max


def tensor_from_pil_image(img, normalize=True):
    return image_resized_to_grid_as_tensor(img, normalize=normalize, multiple_of=1)

def remove_nans(tensor, replace_with=MAX_FLOAT):
    return torch.where(torch.isnan(tensor), replace_with, tensor)


HUE_COLOR_SPACES = [
    "HSV / HSL / RGB",
    "Okhsl",
    "Okhsv",
    "*Oklch / Oklab",
    "*LCh / CIELab",
    "*UPLab (w/CIELab_to_UPLab.icc)",
]


BLEND_MODES = [
    "Normal",
    "Lighten Only",
    "Darken Only",
    "Lighten Only (EAL)",
    "Darken Only (EAL)",
    "Hue",
    "Saturation",
    "Color",
    "Luminosity",
    "Linear Dodge (Add)",
    "Subtract",
    "Multiply",
    "Divide",
    "Screen",
    "Overlay",
    "Linear Burn",
    "Difference",
    "Hard Light",
    "Soft Light",
    "Vivid Light",
    "Linear Light",
    "Color Burn",
    "Color Dodge",
]


BLEND_COLOR_SPACES = [
    "RGB",
    "Linear RGB",
    "HSL (RGB)",
    "HSV (RGB)",
    "Okhsl",
    "Okhsv",
    "Oklch (Oklab)",
    "LCh (CIELab)"
]


@invocation(
    "img_blend",
    title="Image Layer Blend",
    tags=["image", "blend", "layer", "alpha", "composite", "dodge", "burn"],
    category="image",
    version="1.0.11",
)
class ImageBlendInvocation(BaseInvocation):
    """Blend two images together, with optional opacity, mask, and blend modes"""

    layer_upper: ImageField = InputField(description="The top image to blend", ui_order=1)
    blend_mode: Literal[tuple(BLEND_MODES)] = InputField(
        default=BLEND_MODES[0], description="Available blend modes", ui_order=2
    )
    opacity: float = InputField(default=1., description="Desired opacity of the upper layer", ui_order=3)
    mask: Optional[ImageField] = InputField(
        default=None, description="Optional mask, used to restrict areas from blending", ui_order=4
    )
    fit_to_width: bool = InputField(default=False, description="Scale upper layer to fit base width", ui_order=5)
    fit_to_height: bool = InputField(default=True,  description="Scale upper layer to fit base height", ui_order=6)
    layer_base: ImageField = InputField(description="The bottom image to blend", ui_order=7)
    color_space: Literal[tuple(BLEND_COLOR_SPACES)] = InputField(
        default=BLEND_COLOR_SPACES[1], description="Available color spaces for blend computations", ui_order=8
    )
    adaptive_gamut: float = InputField(
        default=0.0,
        description="Adaptive gamut clipping (0=off). Higher prioritizes chroma over lightness", ui_order=9
    )
    high_precision: bool = InputField(
        default=True, description="Use more steps in computing gamut when possible", ui_order=10
    )

    
    def scale_and_pad_or_crop_to_base(self, image_upper, image_base):
        """Rescale upper image based on self.fill_x and self.fill_y params"""

        aspect_base = image_base.width / image_base.height
        aspect_upper = image_upper.width / image_upper.height
        if self.fit_to_width and self.fit_to_height:
            image_upper = image_upper.resize((image_base.width, image_base.height))
        elif ((self.fit_to_width and (aspect_base < aspect_upper)) or
              (self.fit_to_height and (aspect_upper <= aspect_base))):
            image_upper = PIL.ImageOps.pad(image_upper, (image_base.width, image_base.height),
                                           color=(0, 0, 0, 0))
        elif ((self.fit_to_width and (aspect_upper <= aspect_base)) or
              (self.fit_to_height and (aspect_base < aspect_upper))):
            image_upper = PIL.ImageOps.fit(image_upper, (image_base.width, image_base.height))
        return image_upper


    def image_convert_with_xform(self, image_in, from_mode, to_mode):
        """Use PIL ImageCms color management to convert 3-channel image from one mode to another"""

        def fixed_mode(mode):
            if mode.lower() == "srgb":
                return "rgb"
            elif mode.lower() == "cielab":
                return "lab"
            else:
                return mode.lower()

        from_mode, to_mode = fixed_mode(from_mode), fixed_mode(to_mode)

        profile_srgb = None
        profile_uplab = None
        profile_lab = None
        if (from_mode.lower() == "rgb") or (to_mode.lower() == "rgb"):
            profile_srgb = PIL.ImageCms.createProfile("sRGB")
        if (from_mode.lower() == "uplab") or (to_mode.lower() == "uplab"):
            if os.path.isfile("CIELab_to_UPLab.icc"):
                profile_uplab = PIL.ImageCms.getOpenProfile("CIELab_to_UPLab.icc")
        if (from_mode.lower() in ["lab", "cielab", "uplab"]) or  \
           (to_mode.lower() in ["lab", "cielab", "uplab"]):
            if profile_uplab is None:
                profile_lab = PIL.ImageCms.createProfile("LAB", colorTemp=6500)
            else:
                profile_lab = PIL.ImageCms.createProfile("LAB", colorTemp=5000)

        xform_rgb_to_lab = None
        xform_uplab_to_lab = None
        xform_lab_to_uplab = None
        xform_lab_to_rgb = None
        if from_mode == "rgb":
            xform_rgb_to_lab = PIL.ImageCms.buildTransformFromOpenProfiles(
                profile_srgb, profile_lab, "RGB", "LAB", renderingIntent=2, flags=0x2400
            )
        elif from_mode == "uplab":
            xform_uplab_to_lab = PIL.ImageCms.buildTransformFromOpenProfiles(
                profile_uplab, profile_lab, "LAB", "LAB", renderingIntent=2, flags=0x2400
            )
        if to_mode == "uplab":
            xform_lab_to_uplab = PIL.ImageCms.buildTransformFromOpenProfiles(
                profile_lab, profile_uplab, "LAB", "LAB", renderingIntent=2, flags=0x2400
            )
        elif to_mode == "rgb":
            xform_lab_to_rgb = PIL.ImageCms.buildTransformFromOpenProfiles(
                profile_lab, profile_srgb, "LAB", "RGB", renderingIntent=2, flags=0x2400
            )

        image_out = None
        if (from_mode == "rgb") and (to_mode == "lab"):
            image_out = PIL.ImageCms.applyTransform(image_in, xform_rgb_to_lab)
        elif (from_mode == "rgb") and (to_mode == "uplab"):
            image_out = PIL.ImageCms.applyTransform(image_in, xform_rgb_to_lab)
            image_out = PIL.ImageCms.applyTransform(image_out, xform_lab_to_uplab)
        elif (from_mode == "lab") and (to_mode == "uplab"):
            image_out = PIL.ImageCms.applyTransform(image_in, xform_lab_to_uplab)
        elif (from_mode == "lab") and (to_mode == "rgb"):
            image_out = PIL.ImageCms.applyTransform(image_in, xform_lab_to_rgb)
        elif (from_mode == "uplab") and (to_mode == "lab"):
            image_out = PIL.ImageCms.applyTransform(image_in, xform_uplab_to_lab)
        elif (from_mode == "uplab") and (to_mode == "rgb"):
            image_out = PIL.ImageCms.applyTransform(image_in, xform_uplab_to_lab)
            image_out = PIL.ImageCms.applyTransform(image_out, xform_lab_to_rgb)

        return image_out


    def prepare_tensors_from_images(
            self, image_upper, image_lower, mask_image=None,
            required=["hsv", "hsl", "lch", "oklch", "okhsl", "okhsv", "l_eal"]
    ):
        """Convert image to the necessary image space representations for blend calculations"""

        alpha_upper, alpha_lower = None, None
        if (image_upper.mode == "RGBA"):
            # Prepare tensors to compute blend
            image_rgba_upper = image_upper.convert("RGBA")
            alpha_upper = image_rgba_upper.getchannel("A")
            image_upper = image_upper.convert("RGB")
        else:
            if (not (image_upper.mode == "RGB")):
                image_upper = image_upper.convert("RGB")
        if (image_lower.mode == "RGBA"):
            # Prepare tensors to compute blend
            image_rgba_lower = image_lower.convert("RGBA")
            alpha_lower = image_rgba_lower.getchannel("A")
            image_lower = image_lower.convert("RGB")
        else:
            if (not (image_lower.mode == "RGB")):
                image_lower = image_lower.convert("RGB")

        image_lab_upper, image_lab_lower = None, None
        upper_lab_tensor, lower_lab_tensor = None, None
        upper_lch_tensor, lower_lch_tensor = None, None
        if "lch" in required:
            image_lab_upper, image_lab_lower = (
                self.image_convert_with_xform(image_upper, "rgb", "lab"),
                self.image_convert_with_xform(image_lower, "rgb", "lab")
            )

            upper_lab_tensor = torch.stack(
                [
                    tensor_from_pil_image(image_lab_upper.getchannel("L"), normalize=False)[0,:,:],
                    tensor_from_pil_image(image_lab_upper.getchannel("A"), normalize=True)[0,:,:],
                    tensor_from_pil_image(image_lab_upper.getchannel("B"), normalize=True)[0,:,:]
                ]
            )
            lower_lab_tensor = torch.stack(
                [
                    tensor_from_pil_image(image_lab_lower.getchannel("L"), normalize=False)[0,:,:],
                    tensor_from_pil_image(image_lab_lower.getchannel("A"), normalize=True)[0,:,:],
                    tensor_from_pil_image(image_lab_lower.getchannel("B"), normalize=True)[0,:,:]
                ]
            )
            upper_lch_tensor = torch.stack(
                [
                    upper_lab_tensor[0,:,:],
                    torch.sqrt(torch.add(torch.pow(upper_lab_tensor[1,:,:], 2.0),
                                         torch.pow(upper_lab_tensor[2,:,:], 2.0))),
                    torch.atan2(upper_lab_tensor[2,:,:], upper_lab_tensor[1,:,:])
                ]
            )
            lower_lch_tensor = torch.stack(
                [
                    lower_lab_tensor[0,:,:],
                    torch.sqrt(torch.add(torch.pow(lower_lab_tensor[1,:,:], 2.0),
                                         torch.pow(lower_lab_tensor[2,:,:], 2.0))),
                    torch.atan2(lower_lab_tensor[2,:,:], lower_lab_tensor[1,:,:])
                ]
            )

        upper_l_eal_tensor, lower_l_eal_tensor = None, None
        if "l_eal" in required:
            upper_l_eal_tensor = equivalent_achromatic_lightness(upper_lch_tensor)
            lower_l_eal_tensor = equivalent_achromatic_lightness(lower_lch_tensor)

        image_hsv_upper, image_hsv_lower = None, None
        upper_hsv_tensor, lower_hsv_tensor = None, None
        if "hsv" in required:
            image_hsv_upper, image_hsv_lower = image_upper.convert("HSV"), image_lower.convert("HSV")
            upper_hsv_tensor = torch.stack(
                [
                    tensor_from_pil_image(image_hsv_upper.getchannel("H"), normalize=False)[0,:,:],
                    tensor_from_pil_image(image_hsv_upper.getchannel("S"), normalize=False)[0,:,:],
                    tensor_from_pil_image(image_hsv_upper.getchannel("V"), normalize=False)[0,:,:]
                ]
            )
            lower_hsv_tensor = torch.stack(
                [
                    tensor_from_pil_image(image_hsv_lower.getchannel("H"), normalize=False)[0,:,:],
                    tensor_from_pil_image(image_hsv_lower.getchannel("S"), normalize=False)[0,:,:],
                    tensor_from_pil_image(image_hsv_lower.getchannel("V"), normalize=False)[0,:,:]
                ]
            )

        upper_rgb_tensor = tensor_from_pil_image(image_upper, normalize=False)
        lower_rgb_tensor = tensor_from_pil_image(image_lower, normalize=False)

        alpha_upper_tensor, alpha_lower_tensor = None, None
        if alpha_upper is None:
            alpha_upper_tensor = torch.ones(upper_rgb_tensor[0,:,:].shape)
        else:
            alpha_upper_tensor = tensor_from_pil_image(alpha_upper, normalize=False)[0,:,:]
        if alpha_lower is None:
            alpha_lower_tensor = torch.ones(lower_rgb_tensor[0,:,:].shape)
        else:
            alpha_lower_tensor = tensor_from_pil_image(alpha_lower, normalize=False)[0,:,:]

        mask_tensor = None
        if not (mask_image is None):
            mask_tensor = tensor_from_pil_image(mask_image.convert("L"), normalize=False)[0,:,:]

        upper_hsl_tensor, lower_hsl_tensor = None, None
        if "hsl" in required:
            upper_hsl_tensor = hsl_from_srgb(upper_rgb_tensor)
            lower_hsl_tensor = hsl_from_srgb(lower_rgb_tensor)

        upper_okhsl_tensor, lower_okhsl_tensor = None, None
        if "okhsl" in required:
            upper_okhsl_tensor = okhsl_from_srgb(upper_rgb_tensor, steps=(3 if self.high_precision else 1))
            lower_okhsl_tensor = okhsl_from_srgb(lower_rgb_tensor, steps=(3 if self.high_precision else 1))

        upper_okhsv_tensor, lower_okhsv_tensor = None, None
        if "okhsv" in required:
            upper_okhsv_tensor = okhsv_from_srgb(upper_rgb_tensor, steps=(3 if self.high_precision else 1))
            lower_okhsv_tensor = okhsv_from_srgb(lower_rgb_tensor, steps=(3 if self.high_precision else 1))

        upper_rgb_l_tensor = linear_srgb_from_srgb(upper_rgb_tensor)
        lower_rgb_l_tensor = linear_srgb_from_srgb(lower_rgb_tensor)

        upper_oklab_tensor, lower_oklab_tensor = None, None
        upper_oklch_tensor, lower_oklch_tensor = None, None
        if "oklch" in required:
            upper_oklab_tensor = oklab_from_linear_srgb(upper_rgb_l_tensor)
            lower_oklab_tensor = oklab_from_linear_srgb(lower_rgb_l_tensor)

            upper_oklch_tensor = torch.stack(
                [
                    upper_oklab_tensor[0,:,:],
                    torch.sqrt(torch.add(torch.pow(upper_oklab_tensor[1,:,:], 2.0),
                                         torch.pow(upper_oklab_tensor[2,:,:], 2.0))),
                    torch.atan2(upper_oklab_tensor[2,:,:], upper_oklab_tensor[1,:,:])
                ]
            )
            lower_oklch_tensor = torch.stack(
                [
                    lower_oklab_tensor[0,:,:],
                    torch.sqrt(torch.add(torch.pow(lower_oklab_tensor[1,:,:], 2.0),
                                         torch.pow(lower_oklab_tensor[2,:,:], 2.0))),
                    torch.atan2(lower_oklab_tensor[2,:,:], lower_oklab_tensor[1,:,:])
                ]
            )
        
        return (
            upper_rgb_l_tensor,
            lower_rgb_l_tensor,
            upper_rgb_tensor,
            lower_rgb_tensor,
            alpha_upper_tensor,
            alpha_lower_tensor,
            mask_tensor,
            upper_hsv_tensor,
            lower_hsv_tensor,
            upper_hsl_tensor,
            lower_hsl_tensor,
            upper_lab_tensor,
            lower_lab_tensor,
            upper_lch_tensor,
            lower_lch_tensor,
            upper_l_eal_tensor,
            lower_l_eal_tensor,
            upper_oklab_tensor,
            lower_oklab_tensor,
            upper_oklch_tensor,
            lower_oklch_tensor,
            upper_okhsv_tensor,
            lower_okhsv_tensor,
            upper_okhsl_tensor,
            lower_okhsl_tensor,
        )


    def apply_blend(self, image_tensors):
        """Apply the selected blend mode using the appropriate color space representations"""

        blend_mode = self.blend_mode
        color_space = self.color_space.split()[0]
        if (color_space in ["RGB", "Linear"]) and  \
           (blend_mode in ["Hue", "Saturation", "Luminosity", "Color"]):
            color_space = "HSL"

        def adaptive_clipped(rgb_tensor, clamp=True, replace_with=MAX_FLOAT):
            """Keep elements of the tensor finite"""

            rgb_tensor = remove_nans(rgb_tensor, replace_with=replace_with)
            
            if (0 < self.adaptive_gamut):
                rgb_tensor = gamut_clip_tensor(
                    rgb_tensor, alpha=self.adaptive_gamut, steps=(3 if self.high_precision else 1)
                )
                rgb_tensor = remove_nans(rgb_tensor, replace_with=replace_with)
            if clamp:  # Use of MAX_FLOAT seems to lead to NaN's coming back in some cases:
                rgb_tensor = rgb_tensor.clamp(0.,1.)

            return rgb_tensor

        reassembly_function = {
            "RGB": lambda t: linear_srgb_from_srgb(t),
            "Linear": lambda t: t,
            "HSL": lambda t: linear_srgb_from_srgb(srgb_from_hsl(t)),
            "HSV": lambda t: linear_srgb_from_srgb(
                tensor_from_pil_image(
                    pil_image_from_tensor(t.clamp(0., 1.), mode="HSV").convert("RGB"), normalize=False
                )
            ),
            "Okhsl": lambda t: linear_srgb_from_srgb(
                srgb_from_okhsl(t, alpha=self.adaptive_gamut, steps=(3 if self.high_precision else 1))
            ),
            "Okhsv": lambda t: linear_srgb_from_srgb(
                srgb_from_okhsv(t, alpha=self.adaptive_gamut, steps=(3 if self.high_precision else 1))
            ),
            "Oklch": lambda t: linear_srgb_from_oklab(
                torch.stack(
                    [
                        t[0,:,:],
                        torch.mul(t[1,:,:], torch.cos(t[2,:,:])),
                        torch.mul(t[1,:,:], torch.sin(t[2,:,:]))
                    ]
                )
            ),
            "LCh": lambda t: linear_srgb_from_srgb(
                tensor_from_pil_image(
                    self.image_convert_with_xform(
                        PIL.Image.merge("LAB", tuple(map(lambda u: pil_image_from_tensor(u), [
                            t[0,:,:].clamp(0.,1.),
                            torch.div(torch.add(torch.mul(t[1,:,:], torch.cos(t[2,:,:])), 1.), 2.),
                            torch.div(torch.add(torch.mul(t[1,:,:], torch.sin(t[2,:,:])), 1.), 2.)
                        ]))),
                        "lab",
                        "rgb"
                    ),
                    normalize=False
                )
            ),
        }[color_space]

        (
            upper_rgb_l_tensor,  # linear-light sRGB
            lower_rgb_l_tensor,  # linear-light sRGB
            upper_rgb_tensor,
            lower_rgb_tensor,
            alpha_upper_tensor,
            alpha_lower_tensor,
            mask_tensor,
            upper_hsv_tensor, #   h_rgb,   s_hsv,   v_hsv
            lower_hsv_tensor,
            upper_hsl_tensor, #        ,   s_hsl,   l_hsl
            lower_hsl_tensor,
            upper_lab_tensor, #   l_lab,   a_lab,   b_lab
            lower_lab_tensor,
            upper_lch_tensor, #        ,   c_lab,   h_lab
            lower_lch_tensor,
            upper_l_eal_tensor, # l_eal
            lower_l_eal_tensor,
            upper_oklab_tensor, # l_oklab, a_oklab, b_oklab
            lower_oklab_tensor,
            upper_oklch_tensor, #        , c_oklab, h_oklab
            lower_oklch_tensor,
            upper_okhsv_tensor, # h_okhsv, s_okhsv, v_okhsv
            lower_okhsv_tensor,
            upper_okhsl_tensor, # h_okhsl, s_okhsl, l_r_oklab
            lower_okhsl_tensor, 
        ) = image_tensors

        current_space_tensors = {
            "RGB": [upper_rgb_tensor, lower_rgb_tensor],
            "Linear": [upper_rgb_l_tensor, lower_rgb_l_tensor],
            "HSL": [upper_hsl_tensor, lower_hsl_tensor],
            "HSV": [upper_hsv_tensor, lower_hsv_tensor],
            "Okhsl": [upper_okhsl_tensor, lower_okhsl_tensor],
            "Okhsv": [upper_okhsv_tensor, lower_okhsv_tensor],
            "Oklch": [upper_oklch_tensor, lower_oklch_tensor],
            "LCh": [upper_lch_tensor, lower_lch_tensor],
        }[color_space]
        upper_space_tensor = current_space_tensors[0]
        lower_space_tensor = current_space_tensors[1]

        lightness_index = {
            "RGB": None,
            "Linear": None,
            "HSL": 2,
            "HSV": 2,
            "Okhsl": 2,
            "Okhsv": 2,
            "Oklch": 0,
            "LCh": 0,
        }[color_space]

        saturation_index = {
            "RGB": None,
            "Linear": None,
            "HSL": 1,
            "HSV": 1,
            "Okhsl": 1,
            "Okhsv": 1,
            "Oklch": 1,
            "LCh": 1,
        }[color_space]

        hue_index = {
            "RGB": None,
            "Linear": None,
            "HSL": 0,
            "HSV": 0,
            "Okhsl": 0,
            "Okhsv": 0,
            "Oklch": 2,
            "LCh": 2,
        }[color_space]

        if blend_mode == "Normal":
            upper_rgb_l_tensor = reassembly_function(upper_space_tensor)

        elif blend_mode == "Multiply":
            upper_rgb_l_tensor = reassembly_function(
                torch.mul(lower_space_tensor, upper_space_tensor)
            )

        elif blend_mode == "Screen":
            upper_rgb_l_tensor = reassembly_function(
                torch.add(torch.mul(torch.mul(torch.add(torch.mul(upper_space_tensor, -1.), 1.),
                                              torch.add(torch.mul(lower_space_tensor, -1.), 1.)),
                                    -1.),
                          1.)
            )

        elif (blend_mode == "Overlay") or (blend_mode == "Hard Light"):
            subject_of_cond_tensor = (
                lower_space_tensor if (blend_mode == "Overlay") else upper_space_tensor
            )
            if lightness_index is None:
                upper_space_tensor = torch.where(
                    torch.lt(subject_of_cond_tensor, 0.5),
                    torch.mul(torch.mul(lower_space_tensor, upper_space_tensor), 2.),
                    torch.add(
                        torch.mul(
                            torch.mul(
                                torch.mul(torch.add(torch.mul(lower_space_tensor, -1.), 1.),
                                          torch.add(torch.mul(upper_space_tensor, -1.), 1.)),
                                2.
                            ),
                            -1.
                        ),
                        1.
                    )
                )
            else:  # TODO: Currently blending only the lightness channel, not really ideal.
                upper_space_tensor[lightness_index,:,:] = torch.where(
                    torch.lt(subject_of_cond_tensor[lightness_index,:,:], 0.5),
                    torch.mul(torch.mul(lower_space_tensor[lightness_index,:,:],
                                        upper_space_tensor[lightness_index,:,:]), 2.),
                    torch.add(
                        torch.mul(
                            torch.mul(
                                torch.mul(
                                    torch.add(torch.mul(lower_space_tensor[lightness_index,:,:], -1.), 1.),
                                    torch.add(torch.mul(upper_space_tensor[lightness_index,:,:], -1.), 1.)
                                ),
                                2.
                            ),
                            -1.
                        ),
                        1.
                    )
                )
            upper_rgb_l_tensor = adaptive_clipped(reassembly_function(upper_space_tensor))

        elif blend_mode == "Soft Light":
            if lightness_index is None:
                g_tensor = torch.where(
                    torch.le(lower_space_tensor, 0.25),
                    torch.mul(torch.add(torch.mul(torch.sub(torch.mul(lower_space_tensor, 16.), 12.),
                                                  lower_space_tensor), 4.), lower_space_tensor),
                    torch.sqrt(lower_space_tensor)
                )
                lower_space_tensor = torch.where(
                    torch.le(upper_space_tensor, 0.5),
                    torch.sub(
                        lower_space_tensor,
                        torch.mul(
                            torch.mul(
                                torch.add(torch.mul(lower_space_tensor, -1.), 1.),
                                lower_space_tensor
                            ),
                            torch.add(torch.mul(torch.mul(upper_space_tensor, 2.), -1.), 1.)
                        )
                    ),
                    torch.add(
                        lower_space_tensor,
                        torch.mul(
                            torch.sub(torch.mul(upper_space_tensor, 2.), 1.),
                            torch.sub(g_tensor, lower_space_tensor)
                        )
                    )
                )
            else:
                g_tensor = torch.where(  # Calculates all 3 channels but only one is currently used
                    torch.le(
                        lower_space_tensor[lightness_index,:,:], 0.25
                    ).expand(lower_space_tensor.shape),
                    torch.mul(torch.add(torch.mul(torch.sub(torch.mul(lower_space_tensor, 16.), 12.),
                                                  lower_space_tensor), 4.), lower_space_tensor),
                    torch.sqrt(lower_space_tensor)
                )
                lower_space_tensor[lightness_index,:,:] = torch.where(
                    torch.le(
                        upper_space_tensor[lightness_index,:,:], 0.5
                    ),
                    torch.sub(
                        lower_space_tensor[lightness_index,:,:],
                        torch.mul(
                            torch.mul(
                                torch.add(torch.mul(lower_space_tensor[lightness_index,:,:], -1.), 1.),
                                lower_space_tensor[lightness_index,:,:]
                            ),
                            torch.add(torch.mul(torch.mul(upper_space_tensor[lightness_index,:,:], 2.), -1.), 1.)
                        )
                    ),
                    torch.add(
                        lower_space_tensor[lightness_index,:,:],
                        torch.mul(
                            torch.sub(torch.mul(upper_space_tensor[lightness_index,:,:], 2.), 1.),
                            torch.sub(g_tensor, lower_space_tensor[lightness_index,:,:])
                        )
                    )
                )
            upper_rgb_l_tensor = adaptive_clipped(reassembly_function(lower_space_tensor))
        
        elif blend_mode == "Linear Dodge (Add)":
            lower_space_tensor = torch.add(lower_space_tensor, upper_space_tensor)
            if hue_index is not None:
                lower_space_tensor[hue_index,:,:] = torch.remainder(lower_space_tensor[hue_index,:,:], 1.)
            upper_rgb_l_tensor = adaptive_clipped(reassembly_function(lower_space_tensor))

        elif blend_mode == "Color Dodge":
            lower_space_tensor = torch.div(lower_space_tensor, torch.add(torch.mul(upper_space_tensor, -1.), 1.))
            if hue_index is not None:
                lower_space_tensor[hue_index,:,:] = torch.remainder(lower_space_tensor[hue_index,:,:], 1.)
            upper_rgb_l_tensor = adaptive_clipped(reassembly_function(lower_space_tensor))

        elif blend_mode == "Divide":
            lower_space_tensor = torch.div(lower_space_tensor, upper_space_tensor)
            if hue_index is not None:
                lower_space_tensor[hue_index,:,:] = torch.remainder(lower_space_tensor[hue_index,:,:], 1.)
            upper_rgb_l_tensor = adaptive_clipped(reassembly_function(lower_space_tensor))              

        elif blend_mode == "Linear Burn":
            # We compute the result in the lower image's current space tensor and return that:
            if lightness_index is None:  # Elementwise
                lower_space_tensor = torch.sub(
                    torch.add(lower_space_tensor, upper_space_tensor), 1.
                )
            else:  # Operate only on the selected lightness channel
                lower_space_tensor[lightness_index,:,:] = torch.sub(
                    torch.add(lower_space_tensor[lightness_index,:,:],
                              upper_space_tensor[lightness_index,:,:]),
                    1.
                )
            upper_rgb_l_tensor = adaptive_clipped(reassembly_function(lower_space_tensor))

        elif blend_mode == "Color Burn":
            upper_rgb_l_tensor = adaptive_clipped(
                reassembly_function(
                    torch.add(torch.mul(torch.min(torch.div(torch.add(torch.mul(lower_space_tensor, -1.), 1.),
                                                            upper_space_tensor),
                                                  torch.ones(lower_space_tensor.shape)), -1.), 1.)
                )
            )
        elif blend_mode == "Vivid Light":
            if lightness_index is None:
                lower_space_tensor = adaptive_clipped(
                    reassembly_function(
                        torch.where(
                            torch.lt(upper_space_tensor, 0.5),
                            torch.div(
                                torch.add(
                                    torch.mul(
                                        torch.div(
                                            torch.add(torch.mul(lower_space_tensor, -1.), 1.),
                                            upper_space_tensor
                                        ),
                                        -1.
                                    ),
                                    1.
                                ),
                                2.
                            ),
                            torch.div(
                                torch.div(
                                    lower_space_tensor,
                                    torch.add(torch.mul(upper_space_tensor, -1.), 1.)
                                ),
                                2.
                            )
                        )
                    )
                )
            else:
                lower_space_tensor[lightness_index,:,:] = torch.where(
                    torch.lt(upper_space_tensor[lightness_index,:,:], 0.5),
                    torch.div(
                        torch.add(
                            torch.mul(
                                torch.div(
                                    torch.add(torch.mul(lower_space_tensor[lightness_index,:,:], -1.), 1.),
                                    upper_space_tensor[lightness_index,:,:]
                                ),
                                -1.
                            ),
                            1.
                        ),
                        2.
                    ),
                    torch.div(
                        torch.div(
                            lower_space_tensor[lightness_index,:,:],
                            torch.add(torch.mul(upper_space_tensor[lightness_index,:,:], -1.), 1.)
                        ),
                        2.
                    )
                )
            upper_rgb_l_tensor = adaptive_clipped(reassembly_function(lower_space_tensor))
                
        elif blend_mode == "Linear Light":
            if lightness_index is None:
                lower_space_tensor = torch.sub(
                    torch.add(lower_space_tensor, torch.mul(upper_space_tensor, 2.)),
                    1.
                )
            else:
                lower_space_tensor[lightness_index,:,:] = torch.sub(
                    torch.add(lower_space_tensor[lightness_index,:,:],
                              torch.mul(upper_space_tensor[lightness_index,:,:], 2.)),
                    1.
                )
            upper_rgb_l_tensor = adaptive_clipped(reassembly_function(lower_space_tensor))

        elif blend_mode == "Subtract":
            lower_space_tensor = torch.sub(lower_space_tensor, upper_space_tensor)
            if hue_index is not None:
                lower_space_tensor[hue_index,:,:] = torch.remainder(lower_space_tensor[hue_index,:,:], 1.)
            upper_rgb_l_tensor = adaptive_clipped(reassembly_function(lower_space_tensor))

        elif blend_mode == "Difference":
            upper_rgb_l_tensor = adaptive_clipped(
                reassembly_function(
                    torch.abs(torch.sub(lower_space_tensor, upper_space_tensor))
                )
            )

        elif (blend_mode == "Darken Only") or (blend_mode == "Lighten Only"):
            extrema_fn = torch.min if (blend_mode == "Darken Only") else torch.max
            comparator_fn = torch.ge if (blend_mode == "Darken Only") else torch.lt
            if lightness_index is None:
                upper_space_tensor = torch.stack(
                    [
                        extrema_fn(upper_space_tensor[0,:,:], lower_space_tensor[0,:,:]),
                        extrema_fn(upper_space_tensor[1,:,:], lower_space_tensor[1,:,:]),
                        extrema_fn(upper_space_tensor[2,:,:], lower_space_tensor[2,:,:])
                    ]
                )
            else:
                upper_space_tensor = torch.where(
                    comparator_fn(
                        upper_space_tensor[lightness_index,:,:],
                        lower_space_tensor[lightness_index,:,:]
                    ).expand(upper_space_tensor.shape),
                    lower_space_tensor,
                    upper_space_tensor
                )
            upper_rgb_l_tensor = reassembly_function(upper_space_tensor)

        elif blend_mode in ["Hue", "Saturation", "Color", "Luminosity",]:
            if blend_mode == "Hue":  # l, c: lower / h: upper
                upper_space_tensor[lightness_index,:,:] = lower_space_tensor[lightness_index,:,:]
                upper_space_tensor[saturation_index,:,:] = lower_space_tensor[saturation_index,:,:]
            elif blend_mode == "Saturation":  # l, h: lower / c: upper
                upper_space_tensor[lightness_index,:,:] = lower_space_tensor[lightness_index,:,:]
                upper_space_tensor[hue_index,:,:] = lower_space_tensor[hue_index,:,:]
            elif blend_mode == "Color":  # l: lower / c, h: upper
                upper_space_tensor[lightness_index,:,:] = lower_space_tensor[lightness_index,:,:]
            elif blend_mode == "Luminosity":  # h, c: lower / l: upper
                upper_space_tensor[saturation_index,:,:] = lower_space_tensor[saturation_index,:,:]
                upper_space_tensor[hue_index,:,:] = lower_space_tensor[hue_index,:,:]
            upper_rgb_l_tensor = reassembly_function(upper_space_tensor)

        elif blend_mode in ["Lighten Only (EAL)", "Darken Only (EAL)"]:
            comparator_fn = torch.lt if (blend_mode == "Lighten Only (EAL)") else torch.ge
            upper_space_tensor = torch.where(
                comparator_fn(upper_l_eal_tensor,
                              lower_l_eal_tensor).expand(upper_space_tensor.shape),
                lower_space_tensor,
                upper_space_tensor
            )
            upper_rgb_l_tensor = reassembly_function(upper_space_tensor)

        return upper_rgb_l_tensor


    def alpha_composite(
            self,
            upper_tensor,
            alpha_upper_tensor,
            lower_tensor,
            alpha_lower_tensor,
            mask_tensor=None
    ):
        """Alpha compositing of upper on lower tensor with alpha channels, mask and scalar"""

        upper_tensor = remove_nans(upper_tensor)
        
        alpha_upper_tensor = torch.mul(alpha_upper_tensor, self.opacity)
        if not (mask_tensor is None):
            alpha_upper_tensor = torch.mul(alpha_upper_tensor, torch.add(torch.mul(mask_tensor, -1.), 1.))
        
        alpha_tensor = torch.add(
            alpha_upper_tensor,
            torch.mul(alpha_lower_tensor, torch.add(torch.mul(alpha_upper_tensor, -1.), 1.))
        )
        
        return (
            torch.div(torch.add(torch.mul(upper_tensor, alpha_upper_tensor),
                                torch.mul(torch.mul(lower_tensor, alpha_lower_tensor),
                                          torch.add(torch.mul(alpha_upper_tensor, -1.), 1.))),
                      alpha_tensor),
            alpha_tensor
        )


    def invoke(self, context: InvocationContext) -> ImageOutput:
        """Main execution of the ImageBlendInvocation node"""

        image_upper = context.services.images.get_pil_image(self.layer_upper.image_name)
        image_base = context.services.images.get_pil_image(self.layer_base.image_name)

        # Keep the modes for restoration after processing:
        image_mode_upper = image_upper.mode
        image_mode_base = image_base.mode

        # Get rid of ICC profiles by converting to sRGB, but save for restoration:
        cms_profile_srgb = None
        if "icc_profile" in image_upper.info:
            cms_profile_upper = BytesIO(image_upper.info["icc_profile"])
            cms_profile_srgb = PIL.ImageCms.createProfile("sRGB")
            cms_xform = PIL.ImageCms.buildTransformFromOpenProfiles(
                cms_profile_upper, cms_profile_srgb, image_upper.mode, "RGBA"
            )
            image_upper = PIL.ImageCms.applyTransform(image_upper, cms_xform)

        cms_profile_base = None
        icc_profile_bytes = None
        if "icc_profile" in image_base.info:
            icc_profile_bytes = image_base.info["icc_profile"]
            cms_profile_base = BytesIO(icc_profile_bytes)
            if cms_profile_srgb is None:
                cms_profile_srgb = PIL.ImageCms.createProfile("sRGB")
            cms_xform = PIL.ImageCms.buildTransformFromOpenProfiles(
                cms_profile_base, cms_profile_srgb, image_base.mode, "RGBA"
            )
            image_base = PIL.ImageCms.applyTransform(image_base, cms_xform)

        image_mask = None
        if not (self.mask is None):
            image_mask = context.services.images.get_pil_image(self.mask.image_name)
        color_space = self.color_space.split()[0]
        
        image_upper = self.scale_and_pad_or_crop_to_base(
            image_upper, image_base
        )
        if image_mask is not None:
            image_mask = self.scale_and_pad_or_crop_to_base(
                image_mask, image_base
            )

        tensor_requirements = []

        # Hue, Saturation, Color, and Luminosity won't work in sRGB, require HSL
        if self.blend_mode in ["Hue", "Saturation", "Color", "Luminosity"] and  \
           self.color_space in ["RGB", "Linear RGB"]:
            tensor_requirements = ["hsl"]

        if self.blend_mode in ["Lighten Only (EAL)", "Darken Only (EAL)"]:
            tensor_requirements = tensor_requirements + ["lch", "l_eal"]

        tensor_requirements += {
            "Linear": [],
            "RGB": [],
            "HSL": ["hsl"],
            "HSV": ["hsv"],
            "Okhsl": ["okhsl"],
            "Okhsv": ["okhsv"],
            "Oklch": ["oklch"],
            "LCh": ["lch"]
        }[color_space]
            
        image_tensors = (
            upper_rgb_l_tensor,  # linear-light sRGB
            lower_rgb_l_tensor,  # linear-light sRGB
            upper_rgb_tensor,
            lower_rgb_tensor,
            alpha_upper_tensor,
            alpha_lower_tensor,
            mask_tensor,
            upper_hsv_tensor,
            lower_hsv_tensor,
            upper_hsl_tensor,
            lower_hsl_tensor,
            upper_lab_tensor,
            lower_lab_tensor,
            upper_lch_tensor,
            lower_lch_tensor,
            upper_l_eal_tensor,
            lower_l_eal_tensor,
            upper_oklab_tensor,
            lower_oklab_tensor,
            upper_oklch_tensor,
            lower_oklch_tensor,
            upper_okhsv_tensor,
            lower_okhsv_tensor,
            upper_okhsl_tensor,
            lower_okhsl_tensor
        ) = self.prepare_tensors_from_images(
            image_upper, image_base, mask_image=image_mask, required=tensor_requirements
        )

#        if not (self.blend_mode == "Normal"):
        upper_rgb_l_tensor = self.apply_blend(image_tensors)

        output_tensor, alpha_tensor = self.alpha_composite(
            srgb_from_linear_srgb(
                upper_rgb_l_tensor, alpha=self.adaptive_gamut, steps=(3 if self.high_precision else 1)
            ),
            alpha_upper_tensor,
            lower_rgb_tensor,
            alpha_lower_tensor,
            mask_tensor=mask_tensor
        )

        # Restore alpha channel and base mode:
        output_tensor = torch.stack(
            [
                output_tensor[0,:,:],
                output_tensor[1,:,:],
                output_tensor[2,:,:],
                alpha_tensor
            ]
        )
        image_out = pil_image_from_tensor(output_tensor, mode="RGBA")

        # Restore ICC profile if base image had one:
        if not (cms_profile_base is None):
            cms_xform = PIL.ImageCms.buildTransformFromOpenProfiles(
                cms_profile_srgb, BytesIO(icc_profile_bytes), "RGBA", image_out.mode
            )
            image_out = PIL.ImageCms.applyTransform(image_out, cms_xform)
        else:
            image_out = image_out.convert(image_mode_base)
        
        image_dto = context.services.images.create(
            image=image_out,
            image_origin=ResourceOrigin.INTERNAL,
            image_category=ImageCategory.GENERAL,
            node_id=self.id,
            session_id=context.graph_execution_state_id,
            is_intermediate=self.is_intermediate
        )
        return ImageOutput(
            image=ImageField(image_name=image_dto.image_name),
            width=image_dto.width,
            height=image_dto.height
        )


@invocation(
    "img_hue_adjust_plus",
    title="Adjust Image Hue Plus",
    tags=["image", "hue", "oklab", "cielab", "uplab", "lch", "hsv", "hsl", "lab"],
    category="image",
    version="1.0.1",
)
class AdjustImageHuePlusInvocation(BaseInvocation):
    """Adjusts the Hue of an image by rotating it in the selected color space"""

    image: ImageField = InputField(description="The image to adjust")
    space: Literal[tuple(HUE_COLOR_SPACES)] = InputField(
        default=HUE_COLOR_SPACES[1],
        description="Color space in which to rotate hue by polar coords (*: non-invertible)"
    )
    degrees: float = InputField(default=0.0, description="Degrees by which to rotate image hue")
    preserve_lightness: bool = InputField(
        default=False, description="Whether to preserve CIELAB lightness values"
    )
    ok_adaptive_gamut: float = InputField(
        default=0.05, description="Higher preserves chroma at the expense of lightness (Oklab)"
    )
    ok_high_precision: bool = InputField(
        default=True, description="Use more steps in computing gamut (Oklab/Okhsv/Okhsl)"
    )

    def invoke(self, context: InvocationContext) -> ImageOutput:
        image_in = context.services.images.get_pil_image(self.image.image_name)
        image_out = None
        space = self.space.split()[0].lower().strip('*')

        # Keep the mode and alpha channel for restoration after shifting the hue:
        image_mode = image_in.mode
        original_mode = image_mode
        alpha_channel = None
        if (image_mode == "RGBA") or (image_mode == "LA") or (image_mode == "PA"):
            alpha_channel = image_in.getchannel("A")
        elif (image_mode == "RGBa") or (image_mode == "La") or (image_mode == "Pa"):
            alpha_channel = image_in.getchannel("a")
        if (image_mode == "RGBA") or (image_mode == "RGBa"):
            image_mode = "RGB"
        elif (image_mode == "LA") or (image_mode == "La"):
            image_mode = "L"
        elif image_mode == "PA":
            image_mode = "P"

        image_in = image_in.convert("RGB")

        # Keep the CIELAB L* lightness channel for restoration if Preserve Lightness is selected:
        (
            channel_l,
            channel_a,
            channel_b,
            profile_srgb,
            profile_lab,
            profile_uplab,
            lab_transform,
            uplab_transform
        ) = (
            None, None, None, None, None, None, None, None
        )
        if self.preserve_lightness or (space == "lch") or (space == "uplab"):
            profile_srgb = PIL.ImageCms.createProfile("sRGB")
            if space == "uplab":
                if os.path.isfile("CIELab_to_UPLab.icc"):
                    profile_uplab = PIL.ImageCms.getOpenProfile("CIELab_to_UPLab.icc")
            if profile_uplab is None:
                profile_lab = PIL.ImageCms.createProfile("LAB", colorTemp=6500)
            else:
                profile_lab = PIL.ImageCms.createProfile("LAB", colorTemp=5000)

            lab_transform = PIL.ImageCms.buildTransformFromOpenProfiles(
                profile_srgb, profile_lab, "RGB", "LAB", renderingIntent=2, flags=0x2400
            )
            image_out = PIL.ImageCms.applyTransform(image_in, lab_transform)
            if not (profile_uplab is None):
              uplab_transform = PIL.ImageCms.buildTransformFromOpenProfiles(
                  profile_lab, profile_uplab, "LAB", "LAB", renderingIntent=2, flags=0x2400
              )
              image_out = PIL.ImageCms.applyTransform(image_out, uplab_transform)

            channel_l = image_out.getchannel("L")
            channel_a = image_out.getchannel("A")
            channel_b = image_out.getchannel("B")

        if space == "hsv":
            hsv_tensor = tensor_from_pil_image(image_in.convert('HSV'), normalize=False)
            hsv_tensor[0,:,:] = torch.remainder(torch.add(hsv_tensor[0,:,:],
                                                          torch.div(self.degrees, 360.)), 1.)
            image_out = pil_image_from_tensor(hsv_tensor, mode="HSV").convert("RGB")
            
        elif space == "okhsl":
            rgb_tensor = tensor_from_pil_image(image_in.convert("RGB"), normalize=False)
            hsl_tensor = okhsl_from_srgb(rgb_tensor, steps=(3 if self.ok_high_precision else 1))
            hsl_tensor[0,:,:] = torch.remainder(torch.add(hsl_tensor[0,:,:],
                                                          torch.div(self.degrees, 360.)), 1.)
            rgb_tensor = srgb_from_okhsl(hsl_tensor, alpha=0.0)
            image_out = pil_image_from_tensor(rgb_tensor, mode="RGB")

        elif space == "okhsv":
            rgb_tensor = tensor_from_pil_image(image_in.convert("RGB"), normalize=False)
            hsv_tensor = okhsv_from_srgb(rgb_tensor, steps=(3 if self.ok_high_precision else 1))
            hsv_tensor[0,:,:] = torch.remainder(torch.add(hsv_tensor[0,:,:],
                                                          torch.div(self.degrees, 360.)), 1.)
            rgb_tensor = srgb_from_okhsv(hsv_tensor, alpha=0.0)
            image_out = pil_image_from_tensor(rgb_tensor, mode="RGB")

        elif (space == "lch") or (space == "uplab"):
            # <Channels a and b were already extracted, above.>
            
            a_tensor = tensor_from_pil_image(channel_a, normalize=True)
            b_tensor = tensor_from_pil_image(channel_b, normalize=True)

            # L*a*b* to L*C*h
            c_tensor = torch.sqrt(torch.add(torch.pow(a_tensor, 2.0), torch.pow(b_tensor, 2.0)))
            h_tensor = torch.atan2(b_tensor, a_tensor)

            # Rotate h
            rot_rads = (self.degrees / 180.0)*PI

            h_rot = torch.add(h_tensor, rot_rads)
            h_rot = torch.sub(torch.remainder(torch.add(h_rot, PI), 2*PI), PI)

            # L*C*h to L*a*b*
            a_tensor = torch.mul(c_tensor, torch.cos(h_rot))
            b_tensor = torch.mul(c_tensor, torch.sin(h_rot))

            # -1..1 -> 0..1 for all elts of a, b
            a_tensor = torch.div(torch.add(a_tensor, 1.0), 2.0)
            b_tensor = torch.div(torch.add(b_tensor, 1.0), 2.0)

            a_img = pil_image_from_tensor(a_tensor)
            b_img = pil_image_from_tensor(b_tensor)

            image_out = PIL.Image.merge("LAB", (channel_l, a_img, b_img))

            if not (profile_uplab is None):
                deuplab_transform = PIL.ImageCms.buildTransformFromOpenProfiles(
                    profile_uplab, profile_lab, "LAB", "LAB", renderingIntent=2, flags=0x2400
                )
                image_out = PIL.ImageCms.applyTransform(image_out, deuplab_transform)

            rgb_transform = PIL.ImageCms.buildTransformFromOpenProfiles(
                profile_lab, profile_srgb, "LAB", "RGB", renderingIntent=2, flags=0x2400
            )
            image_out = PIL.ImageCms.applyTransform(image_out, rgb_transform)

        elif space == "oklch":
            rgb_tensor = tensor_from_pil_image(image_in.convert("RGB"), normalize=False)

            linear_srgb_tensor = linear_srgb_from_srgb(rgb_tensor)

            lab_tensor = oklab_from_linear_srgb(linear_srgb_tensor)

            # L*a*b* to L*C*h
            c_tensor = torch.sqrt(torch.add(torch.pow(lab_tensor[1,:,:], 2.0),
                                            torch.pow(lab_tensor[2,:,:], 2.0)))
            h_tensor = torch.atan2(lab_tensor[2,:,:], lab_tensor[1,:,:])

            # Rotate h
            rot_rads = (self.degrees / 180.0)*PI

            h_rot = torch.add(h_tensor, rot_rads)
            h_rot = torch.remainder(torch.add(h_rot, 2*PI), 2*PI)

            # L*C*h to L*a*b*
            lab_tensor[1,:,:] = torch.mul(c_tensor, torch.cos(h_rot))
            lab_tensor[2,:,:] = torch.mul(c_tensor, torch.sin(h_rot))

            linear_srgb_tensor = linear_srgb_from_oklab(lab_tensor)

            rgb_tensor = srgb_from_linear_srgb(
                linear_srgb_tensor,
                alpha=self.ok_adaptive_gamut,
                steps=(3 if self.ok_high_precision else 1)
            )
        
            image_out = pil_image_from_tensor(rgb_tensor, mode="RGB")

        # Not all modes can convert directly to LAB using pillow:
        # image_out = image_out.convert("RGB")

        # Restore the L* channel if required:
        if self.preserve_lightness and (not ((space == "lch") or (space == "uplab"))):
            if profile_uplab is None:
                profile_lab = PIL.ImageCms.createProfile("LAB", colorTemp=6500)
            else:
                profile_lab = PIL.ImageCms.createProfile("LAB", colorTemp=5000)

            lab_transform = PIL.ImageCms.buildTransformFromOpenProfiles(
                profile_srgb, profile_lab, "RGB", "LAB", renderingIntent=2, flags=0x2400
            )

            image_out = PIL.ImageCms.applyTransform(image_out, lab_transform)

            if not (profile_uplab is None):
              uplab_transform = PIL.ImageCms.buildTransformFromOpenProfiles(
                  profile_lab, profile_uplab, "LAB", "LAB", renderingIntent=2, flags=0x2400
              )
              image_out = PIL.ImageCms.applyTransform(image_out, uplab_transform)

            image_out = PIL.Image.merge(
                "LAB",
                tuple([channel_l] + [image_out.getchannel(c) for c in "AB"])
            )

            if not (profile_uplab is None):
                deuplab_transform = PIL.ImageCms.buildTransformFromOpenProfiles(
                    profile_uplab, profile_lab, "LAB", "LAB", renderingIntent=2, flags=0x2400
                )
                image_out = PIL.ImageCms.applyTransform(image_out, deuplab_transform)

            rgb_transform = PIL.ImageCms.buildTransformFromOpenProfiles(
                profile_lab, profile_srgb, "LAB", "RGB", renderingIntent=2, flags=0x2400
            )
            image_out = PIL.ImageCms.applyTransform(image_out, rgb_transform)

        # Restore the original image mode, with alpha channel if required:
        image_out = image_out.convert(image_mode)
        if "a" in original_mode.lower():
            image_out = PIL.Image.merge(
                original_mode,
                tuple([image_out.getchannel(c) for c in image_mode] + [alpha_channel])
            )

        image_dto = context.services.images.create(
            image=image_out,
            image_origin=ResourceOrigin.INTERNAL,
            image_category=ImageCategory.GENERAL,
            node_id=self.id,
            session_id=context.graph_execution_state_id,
            is_intermediate=self.is_intermediate
        )
        return ImageOutput(
            image=ImageField(image_name=image_dto.image_name),
            width=image_dto.width,
            height=image_dto.height
        )


def equivalent_achromatic_lightness(lch_tensor):
    """Calculate Equivalent Achromatic Lightness accounting for Helmholtz-Kohlrausch effect"""
    # As described by High, Green, and Nussbaum (2023): https://doi.org/10.1002/col.22839
    
    k = [0.1644, 0.0603, 0.1307, 0.0060]

    h_minus_90 = torch.sub(lch_tensor[2,:,:], PI / 2.0)
    h_minus_90 = torch.sub(torch.remainder(torch.add(h_minus_90, 3*PI), 2*PI), PI)

    f_by = torch.add(k[0] * torch.abs(torch.sin(torch.div(h_minus_90, 2.0))), k[1])
    f_r_0 = torch.add(k[2] * torch.abs(torch.cos(lch_tensor[2,:,:])), k[3])

    f_r = torch.zeros(lch_tensor[0,:,:].shape)
    mask_hi = torch.ge(lch_tensor[2,:,:], -1 * (PI / 2.0))
    mask_lo = torch.le(lch_tensor[2,:,:], PI / 2.0)
    mask = torch.logical_and(mask_hi, mask_lo)
    f_r[mask] = f_r_0[mask]

    l_eal_tensor = torch.add(
        lch_tensor[0,:,:],
        torch.tensordot(torch.add(f_by, f_r), lch_tensor[1,:,:], dims=([0, 1], [0, 1]))
    )
    l_eal_tensor = torch.sub(l_eal_tensor, l_eal_tensor.min())

    return l_eal_tensor


def srgb_from_linear_srgb(linear_srgb_tensor, alpha=0., steps=1):
    """Get gamma-corrected sRGB from a linear-light sRGB image tensor"""

    if 0. < alpha:
        linear_srgb_tensor = gamut_clip_tensor(linear_srgb_tensor, alpha=alpha, steps=steps)
    linear_srgb_tensor = linear_srgb_tensor.clamp(0., 1.)
    mask = torch.lt(linear_srgb_tensor, 0.0404482362771082 / 12.92)
    rgb_tensor = torch.sub(torch.mul(torch.pow(linear_srgb_tensor, (1/2.4)), 1.055), 0.055)
    rgb_tensor[mask] = torch.mul(linear_srgb_tensor[mask], 12.92)

    return rgb_tensor
    

def linear_srgb_from_srgb(srgb_tensor):
    """Get linear-light sRGB from a standard gamma-corrected sRGB image tensor"""

    linear_srgb_tensor = torch.pow(torch.div(torch.add(srgb_tensor, 0.055), 1.055), 2.4)
    linear_srgb_tensor_1 = torch.div(srgb_tensor, 12.92)
    mask = torch.le(srgb_tensor, 0.0404482362771082)
    linear_srgb_tensor[mask] = linear_srgb_tensor_1[mask]

    return linear_srgb_tensor


def max_srgb_saturation_tensor(units_ab_tensor, steps=1):
    """Compute maximum sRGB saturation of a tensor of Oklab ab unit vectors"""
    
    rgb_k_matrix = torch.tensor([[1.19086277,  1.76576728,  0.59662641,  0.75515197, 0.56771245],
                                 [0.73956515, -0.45954494,  0.08285427,  0.12541070, 0.14503204],
                                 [1.35733652, -0.00915799, -1.15130210, -0.50559606, 0.00692167]])
   
    rgb_w_matrix = torch.tensor([[ 4.0767416621, -3.3077115913,  0.2309699292],
                                 [-1.2684380046,  2.6097574011, -0.3413193965],
                                 [-0.0041960863, -0.7034186147,  1.7076147010]])

    rgb_index_firstout_tensor = torch.empty(units_ab_tensor.shape[1:])
    cond_r_tensor = torch.add(torch.mul(-1.88170328, units_ab_tensor[0,:,:]),
                              torch.mul(-0.80936493, units_ab_tensor[1,:,:]))
    cond_g_tensor = torch.add(torch.mul(1.81444104, units_ab_tensor[0,:,:]),
                              torch.mul(-1.19445276, units_ab_tensor[1,:,:]))

    terms_tensor = torch.stack([torch.ones(units_ab_tensor.shape[1:]),
                                units_ab_tensor[0,:,:],
                                units_ab_tensor[1,:,:],
                                torch.pow(units_ab_tensor[0,:,:], 2.),
                                torch.mul(units_ab_tensor[0,:,:],
                                          units_ab_tensor[1,:,:])])

    s_tensor = torch.empty(units_ab_tensor.shape[1:])
    s_tensor = torch.where(
        torch.gt(cond_r_tensor, 1.),
        torch.einsum('twh, t -> wh', terms_tensor, rgb_k_matrix[0]),
        torch.where(torch.gt(cond_g_tensor, 1.),
                    torch.einsum('twh, t -> wh', terms_tensor, rgb_k_matrix[1]),
                    torch.einsum('twh, t -> wh', terms_tensor, rgb_k_matrix[2])))
    
    k_lms_matrix = torch.tensor([[ 0.3963377774,  0.2158037573],
                                 [-0.1055613458, -0.0638541728],
                                 [-0.0894841775, -1.2914855480]])

    k_lms_tensor = torch.einsum('tc, cwh -> twh', k_lms_matrix, units_ab_tensor)

    for i in range(steps):
        root_lms_tensor = torch.add(torch.mul(k_lms_tensor, s_tensor), 1.)
        lms_tensor = torch.pow(root_lms_tensor, 3.)
        lms_ds_tensor = torch.mul(torch.mul(k_lms_tensor, torch.pow(root_lms_tensor, 2.)), 3.)
        lms_ds2_tensor = torch.mul(torch.mul(torch.pow(k_lms_tensor, 2.), root_lms_tensor), 6.)
        f_tensor = torch.where(
            torch.gt(cond_r_tensor, 1.),
            torch.einsum('c, cwh -> wh', rgb_w_matrix[0], lms_tensor),
            torch.where(torch.gt(cond_g_tensor, 1.),
                        torch.einsum('c, cwh -> wh', rgb_w_matrix[1], lms_tensor),
                        torch.einsum('c, cwh -> wh', rgb_w_matrix[2], lms_tensor)))
        f_tensor_1 = torch.where(
            torch.gt(cond_r_tensor, 1.),
            torch.einsum('c, cwh -> wh', rgb_w_matrix[0], lms_ds_tensor),
            torch.where(torch.gt(cond_g_tensor, 1.),
                        torch.einsum('c, cwh -> wh', rgb_w_matrix[1], lms_ds_tensor),
                        torch.einsum('c, cwh -> wh', rgb_w_matrix[2], lms_ds_tensor)))
        f_tensor_2 = torch.where(
            torch.gt(cond_r_tensor, 1.),
            torch.einsum('c, cwh -> wh', rgb_w_matrix[0], lms_ds2_tensor),
            torch.where(torch.gt(cond_g_tensor, 1.),
                        torch.einsum('c, cwh -> wh', rgb_w_matrix[1], lms_ds2_tensor),
                        torch.einsum('c, cwh -> wh', rgb_w_matrix[2], lms_ds2_tensor)))
        s_tensor = torch.sub(s_tensor,
                             torch.div(torch.mul(f_tensor, f_tensor_1),
                                       torch.sub(torch.pow(f_tensor_1, 2.),
                                                 torch.mul(torch.mul(f_tensor, f_tensor_2), 0.5))))

    return s_tensor


def linear_srgb_from_oklab(oklab_tensor):
    """Get linear-light sRGB from an Oklab image tensor"""
    
    # L*a*b* to LMS
    lms_matrix_1 = torch.tensor([[1.,  0.3963377774,  0.2158037573],
                                 [1., -0.1055613458, -0.0638541728],
                                 [1., -0.0894841775, -1.2914855480]])

    lms_tensor_1 = torch.einsum('lwh, kl -> kwh', oklab_tensor, lms_matrix_1)
    lms_tensor = torch.pow(lms_tensor_1, 3.)

    # LMS to linear RGB
    rgb_matrix = torch.tensor([[ 4.0767416621, -3.3077115913,  0.2309699292],
                               [-1.2684380046,  2.6097574011, -0.3413193965],
                               [-0.0041960863, -0.7034186147,  1.7076147010]])

    linear_srgb_tensor = torch.einsum('kwh, sk -> swh', lms_tensor, rgb_matrix)

    return linear_srgb_tensor
    

def oklab_from_linear_srgb(linear_srgb_tensor):
    """Get an Oklab image tensor from a tensor of linear-light sRGB"""
    # linear RGB to LMS
    lms_matrix = torch.tensor([[0.4122214708, 0.5363325363, 0.0514459929],
                               [0.2119034982, 0.6806995451, 0.1073969566],
                               [0.0883024619, 0.2817188376, 0.6299787005]])

    lms_tensor = torch.einsum('cwh, kc -> kwh', linear_srgb_tensor, lms_matrix)

    # LMS to L*a*b*
    lms_tensor_neg_mask = torch.lt(lms_tensor, 0.)
    lms_tensor[lms_tensor_neg_mask] = torch.mul(lms_tensor[lms_tensor_neg_mask], -1.)
    lms_tensor_1 = torch.pow(lms_tensor, 1./3.)
    lms_tensor[lms_tensor_neg_mask] = torch.mul(lms_tensor[lms_tensor_neg_mask], -1.)
    lms_tensor_1[lms_tensor_neg_mask] = torch.mul(lms_tensor_1[lms_tensor_neg_mask], -1.)
    lab_matrix = torch.tensor([[0.2104542553,  0.7936177850, -0.0040720468],
                               [1.9779984951, -2.4285922050,  0.4505937099],
                               [0.0259040371,  0.7827717662, -0.8086757660]])

    lab_tensor = torch.einsum('kwh, lk -> lwh', lms_tensor_1, lab_matrix)

    return lab_tensor


def find_cusp_tensor(units_ab_tensor, steps=1):
    """Compute maximum sRGB lightness and chroma from a tensor of Oklab ab unit vectors"""
    
    s_cusp_tensor = max_srgb_saturation_tensor(units_ab_tensor, steps=steps)

    oklab_tensor = torch.stack([torch.ones(s_cusp_tensor.shape),
                                torch.mul(s_cusp_tensor, units_ab_tensor[0,:,:]),
                                torch.mul(s_cusp_tensor, units_ab_tensor[1,:,:])])

    rgb_at_max_tensor = linear_srgb_from_oklab(oklab_tensor)

    l_cusp_tensor = torch.pow(torch.div(1., rgb_at_max_tensor.max(0).values), 1./3.)
    c_cusp_tensor = torch.mul(l_cusp_tensor, s_cusp_tensor)

    return torch.stack([l_cusp_tensor, c_cusp_tensor])


def find_gamut_intersection_tensor(
        units_ab_tensor,
        l_1_tensor,
        c_1_tensor,
        l_0_tensor,
        steps=1,
        steps_outer=1,
        lc_cusps_tensor=None
):
    """Find thresholds of lightness intersecting RGB gamut from Oklab component tensors"""

    if lc_cusps_tensor is None:
        lc_cusps_tensor = find_cusp_tensor(units_ab_tensor, steps=steps)

    # if (((l_1 - l_0) * c_cusp -
    #      (l_cusp - l_0) * c_1) <= 0.):
    cond_tensor = torch.sub(torch.mul(torch.sub(l_1_tensor, l_0_tensor), lc_cusps_tensor[1,:,:]),
                            torch.mul(torch.sub(lc_cusps_tensor[0,:,:], l_0_tensor), c_1_tensor))
    
    t_tensor = torch.where(
        torch.le(cond_tensor, 0.),  # cond <= 0

        #  t = (c_cusp * l_0) /
        #      ((c_1 * l_cusp) + (c_cusp * (l_0 - l_1)))
        torch.div(torch.mul(lc_cusps_tensor[1,:,:], l_0_tensor),
                  torch.add(torch.mul(c_1_tensor, lc_cusps_tensor[0,:,:]),
                            torch.mul(lc_cusps_tensor[1,:,:],
                                      torch.sub(l_0_tensor, l_1_tensor)))),

        # t = (c_cusp * (l_0-1.)) /
        #     ((c_1 * (l_cusp-1.)) + (c_cusp * (l_0 - l_1)))
        torch.div(torch.mul(lc_cusps_tensor[1,:,:], torch.sub(l_0_tensor, 1.)),
                  torch.add(torch.mul(c_1_tensor, torch.sub(lc_cusps_tensor[0,:,:], 1.)),
                            torch.mul(lc_cusps_tensor[1,:,:],
                                      torch.sub(l_0_tensor, l_1_tensor))))
    )

    for i in range(steps_outer):
        dl_tensor = torch.sub(l_1_tensor, l_0_tensor)
        dc_tensor = c_1_tensor

        k_lms_matrix = torch.tensor([[ 0.3963377774,  0.2158037573],
                                     [-0.1055613458, -0.0638541728],
                                     [-0.0894841775, -1.2914855480]])
        k_lms_tensor = torch.einsum('tc, cwh -> twh', k_lms_matrix, units_ab_tensor)

        lms_dt_tensor = torch.add(torch.mul(k_lms_tensor, dc_tensor), dl_tensor)

        for j in range(steps):

            
            l_tensor = torch.add(torch.mul(l_0_tensor, torch.add(torch.mul(t_tensor, -1.), 1.)),
                                 torch.mul(t_tensor, l_1_tensor))
            c_tensor = torch.mul(t_tensor, c_1_tensor)

            root_lms_tensor = torch.add(torch.mul(k_lms_tensor, c_tensor), l_tensor)

            lms_tensor = torch.pow(root_lms_tensor, 3.)
            lms_dt_tensor_1 = torch.mul(torch.mul(torch.pow(root_lms_tensor, 2.), lms_dt_tensor), 3.)
            lms_dt2_tensor = torch.mul(torch.mul(torch.pow(lms_dt_tensor, 2.), root_lms_tensor), 6.)
            
            rgb_matrix = torch.tensor([[ 4.0767416621, -3.3077115913,  0.2309699292],
                                       [-1.2684380046,  2.6097574011, -0.3413193965],
                                       [-0.0041960863, -0.7034186147,  1.7076147010]])

            rgb_tensor = torch.sub(torch.einsum('qt, twh -> qwh', rgb_matrix, lms_tensor), 1.)
            rgb_tensor_1 = torch.einsum('qt, twh -> qwh', rgb_matrix, lms_dt_tensor_1)
            rgb_tensor_2 = torch.einsum('qt, twh -> qwh', rgb_matrix, lms_dt2_tensor)

            u_rgb_tensor = torch.div(rgb_tensor_1,
                                     torch.sub(torch.pow(rgb_tensor_1, 2.),
                                               torch.mul(torch.mul(rgb_tensor, rgb_tensor_2), 0.5)))

            t_rgb_tensor = torch.mul(torch.mul(rgb_tensor, -1.), u_rgb_tensor)

            max_floats = torch.mul(MAX_FLOAT, torch.ones(t_rgb_tensor.shape))
            
            t_rgb_tensor = torch.where(torch.lt(u_rgb_tensor, 0.), max_floats, t_rgb_tensor)

            t_tensor = torch.where(
                torch.gt(cond_tensor, 0.),
                torch.add(t_tensor, t_rgb_tensor.min(0).values),
                t_tensor
            )
    
    return t_tensor


def gamut_clip_tensor(rgb_l_tensor, alpha=0.05, steps=1, steps_outer=1):
    """Adaptively compress out-of-gamut linear-light sRGB image tensor colors into gamut"""

    lab_tensor = oklab_from_linear_srgb(rgb_l_tensor)
    epsilon = 0.00001
    chroma_tensor = torch.sqrt(
        torch.add(torch.pow(lab_tensor[1,:,:], 2.), torch.pow(lab_tensor[2,:,:], 2.))
    )
    chroma_tensor = torch.where(torch.lt(chroma_tensor, epsilon), epsilon, chroma_tensor)
    
    units_ab_tensor = torch.div(lab_tensor[1:,:,:], chroma_tensor)

    l_d_tensor = torch.sub(lab_tensor[0], 0.5)
    e_1_tensor = torch.add(torch.add(torch.abs(l_d_tensor), torch.mul(chroma_tensor, alpha)), 0.5)
    l_0_tensor = torch.mul(
        torch.add(torch.mul(torch.sign(l_d_tensor),
                            torch.sub(e_1_tensor,
                                      torch.sqrt(torch.sub(torch.pow(e_1_tensor, 2.),
                                                           torch.mul(torch.abs(l_d_tensor), 2.))))),
                  1.),
        0.5)

    t_tensor = find_gamut_intersection_tensor(
        units_ab_tensor,
        lab_tensor[0,:,:],
        chroma_tensor,
        l_0_tensor,
        steps=steps,
        steps_outer=steps_outer
    )
    l_clipped_tensor = torch.add(torch.mul(l_0_tensor, torch.add(torch.mul(t_tensor, -1), 1.)),
                                 torch.mul(t_tensor, lab_tensor[0,:,:]))
    c_clipped_tensor = torch.mul(t_tensor, chroma_tensor)

    return torch.where(torch.logical_or(torch.gt(rgb_l_tensor.max(0).values, 1.),
                                        torch.lt(rgb_l_tensor.min(0).values, 0.)),
                       
                       linear_srgb_from_oklab(torch.stack(
                           [
                               l_clipped_tensor,
                               torch.mul(c_clipped_tensor, units_ab_tensor[0,:,:]),
                               torch.mul(c_clipped_tensor, units_ab_tensor[1,:,:])
                           ]
                       )),

                       rgb_l_tensor)


def st_cusps_from_lc(lc_cusps_tensor):
    """Alternative cusp representation with max C as min(S*L, T*(1-L))"""
    
    return torch.stack(
        [
            torch.div(lc_cusps_tensor[1,:,:], lc_cusps_tensor[0,:,:]),
            torch.div(lc_cusps_tensor[1,:,:], torch.add(torch.mul(lc_cusps_tensor[0,:,:], -1.), 1))
        ]
    )


def ok_l_r_from_l_tensor(x_tensor):
    """Lightness compensated (Y=1) estimate of lightness in Oklab space"""
    
    k_1 = 0.206
    k_2 = 0.03
    k_3 = (1. + k_1) / (1. + k_2)
    #  0.5f * (k_3 * x - k_1 + sqrtf((k_3 * x - k_1) * (k_3 * x - k_1) + 4 * k_2 * k_3 * x));

    return torch.mul(
        torch.add(
            torch.sub(
                torch.mul(x_tensor, k_3),
                k_1),
            torch.sqrt(
                torch.add(
                    torch.pow(torch.sub(torch.mul(x_tensor, k_3), k_1), 2.),
                    torch.mul(torch.mul(torch.mul(x_tensor, k_3), k_2), 4.)
                )
            )
        ),
        0.5
    )


def ok_l_from_lr_tensor(x_tensor):
    """Get uncompensated Oklab lightness from the lightness compensated version"""
    
    k_1 = 0.206
    k_2 = 0.03
    k_3 = (1. + k_1) / (1. + k_2)

    # (x * x + k_1 * x) / (k_3 * (x + k_2))
    return torch.div(
        torch.add(
            torch.pow(x_tensor, 2.),
            torch.mul(x_tensor, k_1)
        ),
        torch.mul(
            torch.add(
                x_tensor,
                k_2
            ),
            k_3
        )        
    )


def srgb_from_okhsv(okhsv_tensor, alpha=0.05, steps=1):
    """Get standard gamma-corrected sRGB from an Okhsv image tensor"""

    okhsv_tensor = okhsv_tensor.clamp(0., 1.)

    units_ab_tensor = torch.stack(
        [
            torch.cos(torch.mul(okhsv_tensor[0,:,:], 2.*PI)),
            torch.sin(torch.mul(okhsv_tensor[0,:,:], 2.*PI))
        ]
    )
    lc_cusps_tensor = find_cusp_tensor(units_ab_tensor, steps=steps)
    st_max_tensor = st_cusps_from_lc(lc_cusps_tensor)
    s_0_tensor = torch.tensor(0.5).expand(st_max_tensor.shape[1:])
    k_tensor = torch.add(torch.mul(torch.div(s_0_tensor, st_max_tensor[0,:,:]), -1.), 1)

    # First compute L and V assuming a perfect triangular gamut
    lc_v_base_tensor = torch.add(s_0_tensor, torch.sub(st_max_tensor[1,:,:],
                                                       torch.mul(st_max_tensor[1,:,:],
                                                                 torch.mul(k_tensor,
                                                                           okhsv_tensor[1,:,:]))))
    lc_v_tensor = torch.stack(
        [
            torch.add(torch.div(torch.mul(torch.mul(okhsv_tensor[1,:,:], s_0_tensor), -1.),
                                lc_v_base_tensor),
                      1.),
            torch.div(torch.mul(torch.mul(okhsv_tensor[1,:,:], st_max_tensor[1,:,:]), s_0_tensor),
                      lc_v_base_tensor)
        ]
    )

    lc_tensor = torch.mul(okhsv_tensor[2,:,:], lc_v_tensor)

    l_vt_tensor = ok_l_from_lr_tensor(lc_v_tensor[0,:,:])
    c_vt_tensor = torch.mul(lc_v_tensor[1,:,:], torch.div(l_vt_tensor, lc_v_tensor[0,:,:]))

    l_new_tensor = ok_l_from_lr_tensor(lc_tensor[0,:,:])
    lc_tensor[1,:,:] = torch.mul(lc_tensor[1,:,:], torch.div(l_new_tensor, lc_tensor[0,:,:]))
    lc_tensor[0,:,:] = l_new_tensor

    rgb_scale_tensor = linear_srgb_from_oklab(
        torch.stack(
            [
                l_vt_tensor,
                torch.mul(units_ab_tensor[0,:,:], c_vt_tensor),
                torch.mul(units_ab_tensor[1,:,:], c_vt_tensor)
            ]
        )
    )

    scale_l_tensor = torch.pow(
        torch.div(1., torch.max(rgb_scale_tensor.max(0).values,
                                torch.zeros(rgb_scale_tensor.shape[1:]))),
        1./3.
    )
    lc_tensor = torch.mul(lc_tensor, scale_l_tensor.expand(lc_tensor.shape))

    rgb_tensor = linear_srgb_from_oklab(
        torch.stack(
            [
                lc_tensor[0,:,:],
                torch.mul(units_ab_tensor[0,:,:], lc_tensor[1,:,:]),
                torch.mul(units_ab_tensor[1,:,:], lc_tensor[1,:,:])
            ]
        )
    )

    rgb_tensor = srgb_from_linear_srgb(rgb_tensor, alpha=alpha, steps=steps)
    return torch.where(torch.isnan(rgb_tensor), 0., rgb_tensor).clamp(0.,1.)


def okhsv_from_srgb(srgb_tensor, steps=1):
    """Get Okhsv image tensor from standard gamma-corrected sRGB"""
    
    lab_tensor = oklab_from_linear_srgb(linear_srgb_from_srgb(srgb_tensor))

    c_tensor = torch.sqrt(torch.add(torch.pow(lab_tensor[1,:,:], 2.),
                                    torch.pow(lab_tensor[2,:,:], 2.)))
    units_ab_tensor = torch.div(lab_tensor[1:,:,:], c_tensor)

    h_tensor = torch.add(torch.div(torch.mul(torch.atan2(torch.mul(lab_tensor[2,:,:], -1.),
                                                         torch.mul(lab_tensor[1,:,:], -1,)),
                                             0.5),
                                   PI),
                         0.5)
    
    lc_cusps_tensor = find_cusp_tensor(units_ab_tensor, steps=steps)
    st_max_tensor = st_cusps_from_lc(lc_cusps_tensor)
    s_0_tensor = torch.tensor(0.5).expand(st_max_tensor.shape[1:])
    k_tensor = torch.add(torch.mul(torch.div(s_0_tensor, st_max_tensor[0,:,:]), -1.), 1)

    t_tensor = torch.div(st_max_tensor[1,:,:],
                         torch.add(c_tensor, torch.mul(lab_tensor[0,:,:], st_max_tensor[1,:,:])))

    l_v_tensor = torch.mul(t_tensor, lab_tensor[0,:,:])
    c_v_tensor = torch.mul(t_tensor, c_tensor)

    l_vt_tensor = ok_l_from_lr_tensor(l_v_tensor)
    c_vt_tensor = torch.mul(c_v_tensor, torch.div(l_vt_tensor, l_v_tensor))

    rgb_scale_tensor = linear_srgb_from_oklab(
        torch.stack(
            [
                l_vt_tensor,
                torch.mul(units_ab_tensor[0,:,:], c_vt_tensor),
                torch.mul(units_ab_tensor[1,:,:], c_vt_tensor)
            ]
        )
    )

    scale_l_tensor = torch.pow(
        torch.div(1., torch.max(rgb_scale_tensor.max(0).values,
                                torch.zeros(rgb_scale_tensor.shape[1:]))),
        1./3.
    )

    lab_tensor[0,:,:] = torch.div(lab_tensor[0,:,:], scale_l_tensor)
    c_tensor = torch.div(c_tensor, scale_l_tensor)

    c_tensor = torch.mul(c_tensor, torch.div(ok_l_r_from_l_tensor(lab_tensor[0,:,:]), lab_tensor[0,:,:]))
    lab_tensor[0,:,:] = ok_l_r_from_l_tensor(lab_tensor[0,:,:])

    v_tensor = torch.div(lab_tensor[0,:,:], l_v_tensor)
    s_tensor = torch.div(torch.mul(torch.add(s_0_tensor, st_max_tensor[1,:,:]), c_v_tensor),
                         torch.add(torch.mul(st_max_tensor[1,:,:], s_0_tensor),
                                   torch.mul(st_max_tensor[1,:,:], torch.mul(k_tensor, c_v_tensor))))

    hsv_tensor = torch.stack([h_tensor, s_tensor, v_tensor])
    return torch.where(torch.isnan(hsv_tensor), 0., hsv_tensor).clamp(0.,1.)


def get_st_mid_tensor(units_ab_tensor):
    """Returns a smooth approximation of cusp, where st_mid < st_max"""

    return torch.stack(
        [
            torch.add(
                torch.div(
                    1.,
                    torch.add(
                        torch.add(
                            torch.mul(units_ab_tensor[1,:,:], 4.15901240),
                            torch.mul(
                                units_ab_tensor[0,:,:],
                                torch.add(
                                    torch.add(
                                        torch.mul(units_ab_tensor[1,:,:], 1.75198401),
                                        torch.mul(
                                            units_ab_tensor[0,:,:],
                                            torch.add(
                                                torch.add(
                                                    torch.mul(units_ab_tensor[1,:,:], -10.02301043),
                                                    torch.mul(
                                                        units_ab_tensor[0,:,:],
                                                        torch.add(
                                                            torch.add(
                                                                torch.mul(units_ab_tensor[1,:,:], 5.38770819),
                                                                torch.mul(units_ab_tensor[0,:,:], 4.69891013)
                                                            ),
                                                            -4.24894561
                                                        )
                                                    )
                                                ),
                                                -2.13704948
                                            )
                                        )
                                    ),
                                    -2.19557347
                                )
                            )
                        ),
                        7.44778970
                    )
                ),
                0.11516993
            ),
            torch.add(
                torch.div(
                    1.,
                    torch.add(
                        torch.add(
                            torch.mul(units_ab_tensor[1,:,:], -0.68124379),
                            torch.mul(
                                units_ab_tensor[0,:,:],
                                torch.add(
                                    torch.add(
                                        torch.mul(units_ab_tensor[1,:,:], 0.90148123),
                                        torch.mul(
                                            units_ab_tensor[0,:,:],
                                            torch.add(
                                                torch.add(
                                                    torch.mul(units_ab_tensor[1,:,:], 0.61223990),
                                                    torch.mul(
                                                        units_ab_tensor[0,:,:],
                                                        torch.add(
                                                            torch.add(
                                                                torch.mul(units_ab_tensor[1,:,:], -0.45399568),
                                                                torch.mul(units_ab_tensor[0,:,:], -0.14661872)
                                                            ),
                                                            0.00299215
                                                        )
                                                    )
                                                ),
                                                -0.27087943
                                            )
                                        )
                                    ),
                                    0.40370612
                                )
                            )
                        ),
                        1.61320320
                    )
                ),
                0.11239642
            )
        ]
    )


def get_cs_tensor(l_tensor, units_ab_tensor, steps=1, steps_outer=1):  # -> [C_0, C_mid, C_max]
    """Arrange minimum, midpoint, and max chroma values from tensors of luminance and ab unit vectors"""

    lc_cusps_tensor = find_cusp_tensor(units_ab_tensor, steps=steps)

    c_max_tensor = find_gamut_intersection_tensor(
        units_ab_tensor,
        l_tensor,
        torch.ones(l_tensor.shape),
        l_tensor,
        lc_cusps_tensor=lc_cusps_tensor,
        steps=steps,
        steps_outer=steps_outer
    )
    st_max_tensor = st_cusps_from_lc(lc_cusps_tensor)
    
    k_tensor = torch.div(c_max_tensor,
                         torch.min(torch.mul(l_tensor, st_max_tensor[0,:,:]),
                                   torch.mul(torch.add(torch.mul(l_tensor, -1.), 1.),
                                             st_max_tensor[1,:,:])))

    st_mid_tensor = get_st_mid_tensor(units_ab_tensor)
    c_a_tensor = torch.mul(l_tensor, st_mid_tensor[0,:,:])
    c_b_tensor = torch.mul(torch.add(torch.mul(l_tensor, -1.), 1.), st_mid_tensor[1,:,:])
    c_mid_tensor = torch.mul(
        torch.mul(k_tensor,
                  torch.sqrt(torch.sqrt(
                      torch.div(1.,
                                torch.add(torch.div(1., torch.pow(c_a_tensor, 4.)),
                                          torch.div(1., torch.pow(c_b_tensor, 4.))))))
                  ),
        0.9
    )

    c_a_tensor = torch.mul(l_tensor, 0.4)
    c_b_tensor = torch.mul(torch.add(torch.mul(l_tensor, -1.), 1.), 0.8)
    c_0_tensor = torch.sqrt(torch.div(1., torch.add(torch.div(1., torch.pow(c_a_tensor, 2.)),
                                                    torch.div(1., torch.pow(c_b_tensor, 2.)))))

    return torch.stack(
        [
            c_0_tensor,
            c_mid_tensor,
            c_max_tensor
        ]
    )


def srgb_from_okhsl(hsl_tensor, alpha=0.05, steps=1, steps_outer=1):
    """Get gamma-corrected sRGB from an Okhsl image tensor"""

    hsl_tensor = hsl_tensor.clamp(0., 1.)

    l_ones_mask = torch.eq(hsl_tensor[2,:,:], 1.)
    l_zeros_mask = torch.eq(hsl_tensor[2,:,:], 0.)
    l_ones_mask = l_ones_mask.expand(hsl_tensor.shape)
    l_zeros_mask = l_zeros_mask.expand(hsl_tensor.shape)
    calc_rgb_mask = torch.logical_not(torch.logical_or(l_ones_mask, l_zeros_mask))

    rgb_tensor = torch.empty(hsl_tensor.shape)
    rgb_tensor = torch.where(l_ones_mask,
                             1.,
                             torch.where(l_zeros_mask,
                                         0.,
                                         rgb_tensor))

    units_ab_tensor = torch.stack(
        [
            torch.cos(torch.mul(hsl_tensor[0,:,:], 2.*PI)),
            torch.sin(torch.mul(hsl_tensor[0,:,:], 2.*PI))
        ]
    )
    l_tensor = ok_l_from_lr_tensor(hsl_tensor[2,:,:])

    # {C_0, C_mid, C_max}    
    cs_tensor = get_cs_tensor(l_tensor, units_ab_tensor, steps=steps, steps_outer=steps_outer)

    mid = 0.8
    mid_inv = 1.25

    s_lt_mid_mask = torch.lt(hsl_tensor[1,:,:], mid)
    t_tensor = torch.where(
        s_lt_mid_mask,
        torch.mul(hsl_tensor[1,:,:], mid_inv),
        torch.div(
            torch.sub(hsl_tensor[1,:,:], mid),
            1. - mid
        )
    )
    k_1_tensor = torch.where(
        s_lt_mid_mask,
        torch.mul(cs_tensor[0,:,:], mid),
        torch.div(
            torch.mul(torch.mul(torch.pow(cs_tensor[1,:,:], 2.), mid_inv**2.), 1. - mid),
            cs_tensor[0,:,:]
        )
    )
    k_2_tensor = torch.where(
        s_lt_mid_mask,
        torch.add(torch.mul(torch.div(k_1_tensor, cs_tensor[1,:,:]), -1.), 1.),
        torch.add(
            torch.mul(torch.div(k_1_tensor, torch.sub(cs_tensor[2,:,:], cs_tensor[1,:,:])), -1.),
            1.
        )
    )

    c_tensor = torch.div(torch.mul(t_tensor, k_1_tensor),
                         torch.add(torch.mul(torch.mul(k_2_tensor, t_tensor), -1.), 1.))
    c_tensor = torch.where(
        s_lt_mid_mask,
        c_tensor,
        torch.add(cs_tensor[1,:,:], c_tensor)
    )

    rgb_tensor = torch.where(
        calc_rgb_mask,
        linear_srgb_from_oklab(
            torch.stack(
                [
                    l_tensor,
                    torch.mul(c_tensor, units_ab_tensor[0,:,:]),
                    torch.mul(c_tensor, units_ab_tensor[1,:,:])
                ]
            )
        ),
        rgb_tensor
    )

    rgb_tensor = srgb_from_linear_srgb(rgb_tensor, alpha=alpha, steps=steps)
    return torch.where(torch.isnan(rgb_tensor), 0., rgb_tensor).clamp(0.,1.)


def okhsl_from_srgb(rgb_tensor, steps=1, steps_outer=1):
    """Get an Okhsl image tensor from gamma-corrected sRGB"""

    lab_tensor = oklab_from_linear_srgb(linear_srgb_from_srgb(rgb_tensor))

    c_tensor = torch.sqrt(
        torch.add(torch.pow(lab_tensor[1,:,:], 2.), torch.pow(lab_tensor[2,:,:], 2.))
    )
    units_ab_tensor = torch.stack([torch.div(lab_tensor[1,:,:], c_tensor),
                                   torch.div(lab_tensor[2,:,:], c_tensor)])

    h_tensor = torch.add(torch.div(torch.mul(torch.atan2(torch.mul(lab_tensor[2,:,:], -1.),
                                                         torch.mul(lab_tensor[1,:,:], -1.)),
                                             0.5),
                                   PI),
                         0.5)

    # {C_0, C_mid, C_max}
    cs_tensor = get_cs_tensor(lab_tensor[0,:,:], units_ab_tensor, steps=1, steps_outer=1)

    mid = 0.8
    mid_inv = 1.25

    c_lt_c_mid_mask = torch.lt(c_tensor, cs_tensor[1,:,:])
    k_1_tensor = torch.where(
        c_lt_c_mid_mask,
        torch.mul(cs_tensor[0,:,:], mid),
        torch.div(torch.mul(torch.mul(torch.pow(cs_tensor[1,:,:], 2.), mid_inv**2), 1. - mid),
                  cs_tensor[0,:,:])
    )
    k_2_tensor = torch.where(
        c_lt_c_mid_mask,
        torch.add(torch.mul(torch.div(k_1_tensor, cs_tensor[1,:,:]), -1.), 1.),
        torch.add(
            torch.mul(torch.div(k_1_tensor, torch.sub(cs_tensor[2,:,:], cs_tensor[1,:,:])), -1.),
            1.
        )
    )
    t_tensor = torch.where(
        c_lt_c_mid_mask,
        torch.div(c_tensor, torch.add(k_1_tensor, torch.mul(k_2_tensor, c_tensor))),
        torch.div(torch.sub(c_tensor, cs_tensor[1,:,:]),
                  torch.add(k_1_tensor, torch.mul(k_2_tensor,
                                                  torch.sub(c_tensor, cs_tensor[1,:,:]))))
    )

    s_tensor = torch.where(
        c_lt_c_mid_mask,
        torch.mul(t_tensor, mid),
        torch.add(torch.mul(t_tensor, 1. - mid), mid)
    )
    l_tensor = ok_l_r_from_l_tensor(lab_tensor[0,:,:])

    hsl_tensor = torch.stack([h_tensor, s_tensor, l_tensor])
    return torch.where(torch.isnan(hsl_tensor), 0., hsl_tensor).clamp(0.,1.)


######################################################################################\
# HSL Code derived from CPython colorsys source code [license text below]
def hsl_from_srgb(rgb_tensor):
    """Get HSL image tensor from standard gamma-corrected sRGB"""
    c_max_tensor = rgb_tensor.max(0).values
    c_min_tensor = rgb_tensor.min(0).values
    c_sum_tensor = torch.add(c_max_tensor, c_min_tensor)
    c_range_tensor = torch.sub(c_max_tensor, c_min_tensor)
    l_tensor = torch.div(c_sum_tensor, 2.0)
    s_tensor = torch.where(
        torch.eq(c_max_tensor, c_min_tensor),
        0.0,
        torch.where(
            torch.lt(l_tensor, 0.5),
            torch.div(c_range_tensor, c_sum_tensor),
            torch.div(c_range_tensor, torch.add(torch.mul(torch.add(c_max_tensor, c_min_tensor), -1.), 2.))
        )
    )
    rgb_c_tensor = torch.div(torch.sub(c_max_tensor.expand(rgb_tensor.shape), rgb_tensor),
                             c_range_tensor.expand(rgb_tensor.shape))
    h_tensor = torch.where(
        torch.eq(c_max_tensor, c_min_tensor),
        0.0,
        torch.where(
            torch.eq(rgb_tensor[0,:,:], c_max_tensor),
            torch.sub(rgb_c_tensor[2,:,:], rgb_c_tensor[1,:,:]),
            torch.where(
                torch.eq(rgb_tensor[1,:,:], c_max_tensor),
                torch.add(torch.sub(rgb_c_tensor[0,:,:], rgb_c_tensor[2,:,:]), 2.),
                torch.add(torch.sub(rgb_c_tensor[1,:,:], rgb_c_tensor[0,:,:]), 4.)
            )
        )
    )
    h_tensor = torch.remainder(torch.div(h_tensor, 6.), 1.)
    return torch.stack([h_tensor, s_tensor, l_tensor])


def srgb_from_hsl(hsl_tensor):
    """Get gamma-corrected sRGB from an HSL image tensor"""
    hsl_tensor = hsl_tensor.clamp(0.,1.)
    rgb_tensor = torch.empty(hsl_tensor.shape)
    s_0_mask = torch.eq(hsl_tensor[1,:,:], 0.)
    s_ne0_mask = torch.logical_not(s_0_mask)
    rgb_tensor = torch.where(
        s_0_mask.expand(rgb_tensor.shape),
        hsl_tensor[2,:,:].expand(hsl_tensor.shape),
        rgb_tensor
    )
    m2_tensor = torch.where(
        torch.le(hsl_tensor[2,:,:], 0.5),
        torch.mul(hsl_tensor[2,:,:], torch.add(hsl_tensor[1,:,:], 1.)),
        torch.sub(torch.add(hsl_tensor[2,:,:], hsl_tensor[1,:,:]),
                  torch.mul(hsl_tensor[2,:,:], hsl_tensor[1,:,:]))
    )
    m1_tensor = torch.sub(torch.mul(hsl_tensor[2,:,:], 2.), m2_tensor)

    def hsl_values(m1_tensor, m2_tensor, h_tensor):
        """Helper for computing output components"""

        h_tensor = torch.remainder(h_tensor, 1.)
        result_tensor = m1_tensor.clone()
        result_tensor = torch.where(
            torch.lt(h_tensor, 1./6.),
            torch.add(m1_tensor,
                      torch.mul(torch.sub(m2_tensor, m1_tensor),
                                torch.mul(h_tensor, 6.))),
            torch.where(
                torch.lt(h_tensor, 0.5),
                m2_tensor,
                torch.where(
                    torch.lt(h_tensor, 2./3.),
                    torch.add(m1_tensor,
                              torch.mul(
                                  torch.sub(m2_tensor, m1_tensor),
                                  torch.mul(torch.add(torch.mul(h_tensor, -1.), 2./3.),
                                            6.)
                              )
                    ),
                    result_tensor
                )
            )
        )
        return result_tensor

    return torch.stack(
        [
            hsl_values(m1_tensor, m2_tensor, torch.add(hsl_tensor[0,:,:], 1./3.)),
            hsl_values(m1_tensor, m2_tensor, hsl_tensor[0,:,:]),
            hsl_values(m1_tensor, m2_tensor, torch.sub(hsl_tensor[0,:,:], 1./3.))
        ]
    )

# PSF LICENSE AGREEMENT FOR PYTHON 3.11.5

# 1. This LICENSE AGREEMENT is between the Python Software Foundation ("PSF"), and
#    the Individual or Organization ("Licensee") accessing and otherwise using Python
#    3.11.5 software in source or binary form and its associated documentation.

# 2. Subject to the terms and conditions of this License Agreement, PSF hereby
#    grants Licensee a nonexclusive, royalty-free, world-wide license to reproduce,
#    analyze, test, perform and/or display publicly, prepare derivative works,
#    distribute, and otherwise use Python 3.11.5 alone or in any derivative
#    version, provided, however, that PSF's License Agreement and PSF's notice of
#    copyright, i.e., "Copyright (c) 2001-2023 Python Software Foundation; All Rights
#    Reserved" are retained in Python 3.11.5 alone or in any derivative version
#    prepared by Licensee.

# 3. In the event Licensee prepares a derivative work that is based on or
#    incorporates Python 3.11.5 or any part thereof, and wants to make the
#    derivative work available to others as provided herein, then Licensee hereby
#    agrees to include in any such work a brief summary of the changes made to Python
#    3.11.5.

# 4. PSF is making Python 3.11.5 available to Licensee on an "AS IS" basis.
#    PSF MAKES NO REPRESENTATIONS OR WARRANTIES, EXPRESS OR IMPLIED.  BY WAY OF
#    EXAMPLE, BUT NOT LIMITATION, PSF MAKES NO AND DISCLAIMS ANY REPRESENTATION OR
#    WARRANTY OF MERCHANTABILITY OR FITNESS FOR ANY PARTICULAR PURPOSE OR THAT THE
#    USE OF PYTHON 3.11.5 WILL NOT INFRINGE ANY THIRD PARTY RIGHTS.

# 5. PSF SHALL NOT BE LIABLE TO LICENSEE OR ANY OTHER USERS OF PYTHON 3.11.5
#    FOR ANY INCIDENTAL, SPECIAL, OR CONSEQUENTIAL DAMAGES OR LOSS AS A RESULT OF
#    MODIFYING, DISTRIBUTING, OR OTHERWISE USING PYTHON 3.11.5, OR ANY DERIVATIVE
#    THEREOF, EVEN IF ADVISED OF THE POSSIBILITY THEREOF.

# 6. This License Agreement will automatically terminate upon a material breach of
#    its terms and conditions.

# 7. Nothing in this License Agreement shall be deemed to create any relationship
#    of agency, partnership, or joint venture between PSF and Licensee.  This License
#    Agreement does not grant permission to use PSF trademarks or trade name in a
#    trademark sense to endorse or promote products or services of Licensee, or any
#    third party.

# 8. By copying, installing or otherwise using Python 3.11.5, Licensee agrees
#    to be bound by the terms and conditions of this License Agreement.
######################################################################################/
