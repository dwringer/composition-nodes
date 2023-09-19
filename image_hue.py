# Copyright (c) 2023 Darren Ringer <dwringer@gmail.com>
# Parts based on Oklab: Copyright (c) 2021 Björn Ottosson <https://bottosson.github.io/>

import os.path
from math import pi as PI
from typing import Literal

import PIL.Image
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


def tensor_from_pil_image(img, normalize=True):
    return image_resized_to_grid_as_tensor(img, normalize=normalize, multiple_of=1)


MAX_FLOAT = torch.finfo(torch.tensor(1.).dtype).max

COLOR_SPACES = [
    "HSV / HSL / RGB",
    "Okhsl",
    "Okhsv",
    "*Oklch / Oklab",
    "*LCh / CIELab",
    "*UPLab (w/CIELab_to_UPLab.icc)",
]


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
    space: Literal[tuple(COLOR_SPACES)] = InputField(
        default=COLOR_SPACES[1],
        description="Color space in which to rotate hue by polar coords (*: non-invertible)"
    )
    degrees: float = InputField(default=0.0, description="Degrees by which to rotate image hue")
    preserve_lightness: bool = InputField(
        default=False, description="Whether to preserve CIELAB lightness values"
    )
    ok_adaptive_gamut: float = InputField(
        default=0.5, description="Lower preserves lightness at the expense of chroma (Oklab)"
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
            rgb_tensor = srgb_from_okhsl(hsl_tensor, alpha=(0.05 if self.ok_high_precision else 0.0))
            image_out = pil_image_from_tensor(rgb_tensor, mode="RGB")

        elif space == "okhsv":
            rgb_tensor = tensor_from_pil_image(image_in.convert("RGB"), normalize=False)
            hsv_tensor = okhsv_from_srgb(rgb_tensor, steps=(3 if self.ok_high_precision else 1))
            hsv_tensor[0,:,:] = torch.remainder(torch.add(hsv_tensor[0,:,:],
                                                          torch.div(self.degrees, 360.)), 1.)
            rgb_tensor = srgb_from_okhsv(hsv_tensor, alpha=(0.05 if self.ok_high_precision else 0.0))
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


def srgb_from_linear_srgb(linear_srgb_tensor, alpha=0.05, steps=1):
    if 0 < alpha:
        linear_srgb_tensor = gamut_clip_tensor(linear_srgb_tensor, alpha=alpha, steps=steps)
    linear_srgb_tensor = linear_srgb_tensor.clamp(0., 1.)
    mask = torch.lt(linear_srgb_tensor, 0.0404482362771082 / 12.92)
    rgb_tensor = torch.sub(torch.mul(torch.pow(linear_srgb_tensor, (1/2.4)), 1.055), 0.055)
    rgb_tensor[mask] = torch.mul(linear_srgb_tensor[mask], 12.92)

    return rgb_tensor
    

def linear_srgb_from_srgb(srgb_tensor):
    linear_srgb_tensor = torch.pow(torch.div(torch.add(srgb_tensor, 0.055), 1.055), 2.4)
    linear_srgb_tensor_1 = torch.div(srgb_tensor, 12.92)
    mask = torch.le(srgb_tensor, 0.0404482362771082)
    linear_srgb_tensor[mask] = linear_srgb_tensor_1[mask]

    return linear_srgb_tensor


def max_srgb_saturation_tensor(units_ab_tensor, steps=1):
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


def gamut_clip_tensor(rgb_tensor, alpha=0.05, steps=1, steps_outer=1):
    lab_tensor = oklab_from_linear_srgb(rgb_tensor)
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

    return torch.where(torch.logical_or(torch.gt(rgb_tensor.max(0).values, 1.),
                                        torch.lt(rgb_tensor.min(0).values, 0.)),
                       
                       linear_srgb_from_oklab(torch.stack(
                           [
                               l_clipped_tensor,
                               torch.mul(c_clipped_tensor, units_ab_tensor[0,:,:]),
                               torch.mul(c_clipped_tensor, units_ab_tensor[1,:,:])
                           ]
                       )),

                       rgb_tensor)


def st_cusps_from_lc(lc_cusps_tensor):
    return torch.stack(
        [
            torch.div(lc_cusps_tensor[1,:,:], lc_cusps_tensor[0,:,:]),
            torch.div(lc_cusps_tensor[1,:,:], torch.add(torch.mul(lc_cusps_tensor[0,:,:], -1.), 1))
        ]
    )


def toe(x_tensor):
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


def toe_inverse(x_tensor):
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

    l_vt_tensor = toe_inverse(lc_v_tensor[0,:,:])
    c_vt_tensor = torch.mul(lc_v_tensor[1,:,:], torch.div(l_vt_tensor, lc_v_tensor[0,:,:]))

    l_new_tensor = toe_inverse(lc_tensor[0,:,:])
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

    return srgb_from_linear_srgb(rgb_tensor, alpha=alpha, steps=steps)


def okhsv_from_srgb(srgb_tensor, steps=1):
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

    l_vt_tensor = toe_inverse(l_v_tensor)
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

    c_tensor = torch.mul(c_tensor, torch.div(toe(lab_tensor[0,:,:]), lab_tensor[0,:,:]))
    lab_tensor[0,:,:] = toe(lab_tensor[0,:,:])

    v_tensor = torch.div(lab_tensor[0,:,:], l_v_tensor)
    s_tensor = torch.div(torch.mul(torch.add(s_0_tensor, st_max_tensor[1,:,:]), c_v_tensor),
                         torch.add(torch.mul(st_max_tensor[1,:,:], s_0_tensor),
                                   torch.mul(st_max_tensor[1,:,:], torch.mul(k_tensor, c_v_tensor))))

    return torch.stack([h_tensor, s_tensor, v_tensor])


def get_st_mid_tensor(units_ab_tensor):
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
    l_tensor = toe_inverse(hsl_tensor[2,:,:])

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

    return srgb_from_linear_srgb(rgb_tensor, alpha=alpha, steps=steps)


def okhsl_from_srgb(rgb_tensor, steps=1, steps_outer=1):
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
    l_tensor = toe(lab_tensor[0,:,:])

    return torch.stack([h_tensor, s_tensor, l_tensor])
