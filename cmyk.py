import math
import os
import os.path
from typing import Literal, Optional

import numpy
from PIL import Image, ImageCms

from invokeai.app.invocations.baseinvocation import (
    BaseInvocation,
    BaseInvocationOutput,
    InputField,
    InvocationContext,
    OutputField,
    WithMetadata,
    invocation,
    invocation_output,
)
from invokeai.app.invocations.primitives import ImageField, ImageOutput
from invokeai.app.services.image_records.image_records_common import ImageCategory, ResourceOrigin

COLOR_PROFILES_DIR = "nodes/color-profiles/"


def load_profiles() -> list:
    """Load available ICC profile filenames from COLOR_PROFILES_DIR into a dict"""

    path = COLOR_PROFILES_DIR.strip('/')
    profiles = {
        "Default": "Default",
        "PIL": "PIL"
    }
    extensions = [".icc", ".icm"]

    if os.path.exists(path):
        for icc_filename in os.listdir(path):
            if icc_filename[-4:].lower() in extensions:
                profile = ImageCms.getOpenProfile(
                    path + '/' + icc_filename
                ).profile
                description = profile.profile_description
                desc_ext = description[-4:].lower()  \
                    if (description[-4:].lower() in extensions)  \
                       else None
                manufacturer = profile.manufacturer
                model = profile.model

                if manufacturer is None:
                    manufacturer = profile.header_manufacturer
                if manufacturer is not None:
                    if manufacturer.isascii() and  \
                       (not (len(manufacturer.strip('\x00')) == 0)):
                        manufacturer = manufacturer.title()
                    else:
                        manufacturer = None

                name = None
                if ((manufacturer is None) and (model is None)) or  \
                   ((manufacturer is not None or (model is None)) and (desc_ext is None)):
                    if desc_ext is None:
                        name = description
                    else:
                        name = description[:-4]
                    name = name.replace('_', ' ')
                elif manufacturer is None:
                    name = model.replace('_', ' ') + "(" + icc_filename + ")"
                elif model is None:
                    name = manufacturer + " : " + '.'.join(icc_filename.split('.')[:-1])
                else:
                    name = manufacturer + " : " + model.replace('_', ' ')

                profiles[name] = icc_filename

    return profiles


color_profiles: list = load_profiles()


@invocation_output("cmyk_split_output")
class CMYKSplitOutput(BaseInvocationOutput):
    """Base class for invocations that output four L-mode images (C, M, Y, K)"""
    c_channel: ImageField = OutputField(description="Grayscale image of the cyan channel")
    m_channel: ImageField = OutputField(description="Grayscale image of the magenta channel")
    y_channel: ImageField = OutputField(description="Grayscale image of the yellow channel")
    k_channel: ImageField = OutputField(description="Grayscale image of the k channel")
    alpha_channel: ImageField = OutputField(description="Grayscale image of the alpha channel")
    width: int = OutputField(description="The width of the image in pixels")
    height: int = OutputField(description="The height of the image in pixels")


@invocation("cmyk_split", title="CMYK Split", tags=["cmyk", "image", "color"], category="image", version="1.1.0")
class CMYKSplitInvocation(BaseInvocation, WithMetadata):
    """Split an image into subtractive color channels (CMYK+alpha)"""

    image: ImageField = InputField(description="The image to halftone", default=None)
    profile: Literal[tuple(color_profiles.keys())] = InputField(
        default=list(color_profiles.keys())[0], description="CMYK Color Profile"
    )

    def pil_from_array(self, arr):
        return Image.fromarray((arr * 255).astype("uint8"))

    def array_from_pil(self, img):
        return numpy.array(img) / 255

    def convert_rgb_to_cmyk(self, image: Image) -> Image:
        if self.profile == "Default":
            r = self.array_from_pil(image.getchannel("R"))
            g = self.array_from_pil(image.getchannel("G"))
            b = self.array_from_pil(image.getchannel("B"))

            k = 1 - numpy.maximum(numpy.maximum(r, g), b)
            c = (1 - r - k) / (1 - k)
            m = (1 - g - k) / (1 - k)
            y = (1 - b - k) / (1 - k)

            c = self.pil_from_array(c)
            m = self.pil_from_array(m)
            y = self.pil_from_array(y)
            k = self.pil_from_array(k)

            return Image.merge("CMYK", (c, m, y, k))
        elif self.profile == "PIL":
            return image.convert("CMYK")
        else:
            image_rgb = image.convert("RGB")
            cms_profile_cmyk = ImageCms.getOpenProfile(
                COLOR_PROFILES_DIR + color_profiles[self.profile]
            )
            cms_profile_srgb = ImageCms.createProfile("sRGB")
            cms_xform = ImageCms.buildTransformFromOpenProfiles(
                cms_profile_srgb, cms_profile_cmyk, "RGB", "CMYK",
                renderingIntent=ImageCms.Intent.RELATIVE_COLORIMETRIC,
                flags=(ImageCms.FLAGS["BLACKPOINTCOMPENSATION"] |
                       ImageCms.FLAGS["HIGHRESPRECALC"]),
            )
            return ImageCms.applyTransform(image_rgb, cms_xform)

    def invoke(self, context: InvocationContext) -> CMYKSplitOutput:
        image = context.services.images.get_pil_image(self.image.image_name)
        mode = image.mode
        width, height = image.size

        alpha_channel = image.getchannel("A") if mode == "RGBA" else Image.new("L", image.size, color=255)

        image = self.convert_rgb_to_cmyk(image)

        c, m, y, k = image.split()

        image_c_dto = context.services.images.create(
            image=Image.fromarray(numpy.array(c)),
            image_origin=ResourceOrigin.INTERNAL,
            image_category=ImageCategory.GENERAL,
            node_id=self.id,
            session_id=context.graph_execution_state_id,
            is_intermediate=self.is_intermediate,
            metadata=self.metadata,
            workflow=context.workflow,
        )
        image_m_dto = context.services.images.create(
            image=Image.fromarray(numpy.array(m)),
            image_origin=ResourceOrigin.INTERNAL,
            image_category=ImageCategory.GENERAL,
            node_id=self.id,
            session_id=context.graph_execution_state_id,
            is_intermediate=self.is_intermediate,
            metadata=self.metadata,
            workflow=context.workflow,
        )
        image_y_dto = context.services.images.create(
            image=Image.fromarray(numpy.array(y)),
            image_origin=ResourceOrigin.INTERNAL,
            image_category=ImageCategory.GENERAL,
            node_id=self.id,
            session_id=context.graph_execution_state_id,
            is_intermediate=self.is_intermediate,
            metadata=self.metadata,
            workflow=context.workflow,
        )
        image_k_dto = context.services.images.create(
            image=Image.fromarray(numpy.array(k)),
            image_origin=ResourceOrigin.INTERNAL,
            image_category=ImageCategory.GENERAL,
            node_id=self.id,
            session_id=context.graph_execution_state_id,
            is_intermediate=self.is_intermediate,
            metadata=self.metadata,
            workflow=context.workflow,
        )
        image_alpha_dto = context.services.images.create(
            image=Image.fromarray(numpy.array(alpha_channel)),
            image_origin=ResourceOrigin.INTERNAL,
            image_category=ImageCategory.GENERAL,
            node_id=self.id,
            session_id=context.graph_execution_state_id,
            is_intermediate=self.is_intermediate,
            metadata=self.metadata,
            workflow=context.workflow,
        )

        return CMYKSplitOutput(
            c_channel=ImageField(image_name=image_c_dto.image_name),
            m_channel=ImageField(image_name=image_m_dto.image_name),
            y_channel=ImageField(image_name=image_y_dto.image_name),
            k_channel=ImageField(image_name=image_k_dto.image_name),
            alpha_channel=ImageField(image_name=image_alpha_dto.image_name),
            width=image.width,
            height=image.height,
        )


@invocation("cmyk_merge", title="CMYK Merge", tags=["cmyk", "image", "color"], category="image", version="1.1.0")
class CMYKMergeInvocation(BaseInvocation, WithMetadata):
    """Merge subtractive color channels (CMYK+alpha)"""

    c_channel: Optional[ImageField] = InputField(description="The c channel", default=None)
    m_channel: Optional[ImageField] = InputField(description="The m channel", default=None)
    y_channel: Optional[ImageField] = InputField(description="The y channel", default=None)
    k_channel: Optional[ImageField] = InputField(description="The k channel", default=None)
    alpha_channel: Optional[ImageField] = InputField(description="The alpha channel", default=None)
    profile: Literal[tuple(color_profiles.keys())] = InputField(
        default=list(color_profiles.keys())[0], description="CMYK Color Profile"
    )

    def pil_from_array(self, arr):
        return Image.fromarray((arr * 255).astype("uint8"))

    def array_from_pil(self, img):
        return numpy.array(img) / 255

    def convert_cmyk_to_rgb(self, image: Image) -> Image:
        if self.profile == "Default":
            c = self.array_from_pil(image.getchannel("C"))
            m = self.array_from_pil(image.getchannel("M"))
            y = self.array_from_pil(image.getchannel("Y"))
            k = self.array_from_pil(image.getchannel("K"))

            r = (1 - c) * (1 - k)
            g = (1 - m) * (1 - k)
            b = (1 - y) * (1 - k)

            r = self.pil_from_array(r)
            g = self.pil_from_array(g)
            b = self.pil_from_array(b)

            return Image.merge("RGB", (r, g, b))
        elif self.profile == "PIL":
            return image.convert("RGB")
        else:
            cms_profile_cmyk = ImageCms.getOpenProfile(
                COLOR_PROFILES_DIR + color_profiles[self.profile]
            )
            cms_profile_srgb = ImageCms.createProfile("sRGB")
            cms_xform = ImageCms.buildTransformFromOpenProfiles(
                cms_profile_cmyk, cms_profile_srgb, "CMYK", "RGB",
                renderingIntent=ImageCms.Intent.RELATIVE_COLORIMETRIC,
                flags=(ImageCms.FLAGS["BLACKPOINTCOMPENSATION"] |
                       ImageCms.FLAGS["HIGHRESPRECALC"]),
            )
            return ImageCms.applyTransform(image, cms_xform)


    def invoke(self, context: InvocationContext) -> ImageOutput:
        c_image, m_image, y_image, k_image, alpha_image = None, None, None, None, None
        if self.c_channel is not None:
            c_image = context.services.images.get_pil_image(self.c_channel.image_name)
            c_image = c_image.convert("L")
        if self.m_channel is not None:
            m_image = context.services.images.get_pil_image(self.m_channel.image_name)
            m_image = m_image.convert("L")
        if self.y_channel is not None:
            y_image = context.services.images.get_pil_image(self.y_channel.image_name)
            y_image = y_image.convert("L")
        if self.k_channel is not None:
            k_image = context.services.images.get_pil_image(self.k_channel.image_name)
            k_image = k_image.convert("L")
        if self.alpha_channel is not None:
            alpha_image = context.services.images.get_pil_image(self.alpha_channel.image_name)
            alpha_image = alpha_image.convert("L")

        image_size = None
        i = 0
        images = [c_image, m_image, y_image, k_image, alpha_image]
        while image_size is None and i < len(images):
            if images[i] is not None:
                image_size = images[i].size
            i += 1

        if c_image is None:
            c_image =  Image.new("L", image_size, color=0)
        if m_image is None:
            m_image =  Image.new("L", image_size, color=0)
        if y_image is None:
            y_image = Image.new("L", image_size, color=0)
        if k_image is None:
            k_image = Image.new("L", image_size, color=0)

        cmyk = Image.merge("CMYK", (c_image, m_image, y_image, k_image))

        image = self.convert_cmyk_to_rgb(cmyk)

        if alpha_image is not None:
            image = image.convert("RGBA")
            image.putalpha(alpha_image)
        image = Image.fromarray(numpy.array(image), mode=image.mode)

        image_dto = context.services.images.create(
            image=image,
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
            width=image.width,
            height=image.height,
        )


@invocation_output("cmyk_separation_output")
class CMYKSeparationOutput(BaseInvocationOutput):
    """Base class for invocations that output four L-mode images (C, M, Y, K)"""
    color_image: ImageField = OutputField(description="Blank image of the specified color")
    width: int = OutputField(description="The width of the image in pixels")
    height: int = OutputField(description="The height of the image in pixels")
    part_a: ImageField = OutputField(description="Blank image of the first separated color")
    rgb_red_a: int = OutputField(description="R value of color part A")
    rgb_green_a: int = OutputField(description="G value of color part A")
    rgb_blue_a: int = OutputField(description="B value of color part A")
    part_b: ImageField = OutputField(description="Blank image of the second separated color")
    rgb_red_b: int = OutputField(description="R value of color part B")
    rgb_green_b: int = OutputField(description="G value of color part B")
    rgb_blue_b: int = OutputField(description="B value of color part B")


@invocation("cmyk_separation", title="CMYK Color Separation", tags=["image", "cmyk", "separation", "color"], category="image", version="1.1.0")
class CMYKColorSeparationInvocation(BaseInvocation, WithMetadata):
    """Get color images from a base color and two others that subtractively mix to obtain it"""
    width: int = InputField(default=512, description="Desired image width")
    height: int = InputField(default=512, description="Desired image height")
    c_value: float = InputField(default=0, description="Desired final cyan value")
    m_value: float = InputField(default=25, description="Desired final magenta value")
    y_value: float = InputField(default=28, description="Desired final yellow value")
    k_value: float = InputField(default=76, description="Desired final black value")
    c_split: float = InputField(default=.5, description="Desired cyan split point % [0..1.0]")
    m_split: float = InputField(default=1., description="Desired magenta split point % [0..1.0]")
    y_split: float = InputField(default=0., description="Desired yellow split point % [0..1.0]")
    k_split: float = InputField(default=0.5, description="Desired black split point % [0..1.0]")
    profile: Literal[tuple(color_profiles.keys())] = InputField(
        default=list(color_profiles.keys())[0], description="CMYK Color Profile"
    )

    def pil_from_array(self, arr):
        return Image.fromarray((arr * 255).astype("uint8"))

    def array_from_pil(self, img):
        return numpy.array(img) / 255

    def convert_cmyk_to_rgb(self, image: Image) -> Image:
        if self.profile == "Default":
            c = self.array_from_pil(image.getchannel("C"))
            m = self.array_from_pil(image.getchannel("M"))
            y = self.array_from_pil(image.getchannel("Y"))
            k = self.array_from_pil(image.getchannel("K"))

            r = (1 - c) * (1 - k)
            g = (1 - m) * (1 - k)
            b = (1 - y) * (1 - k)

            r = self.pil_from_array(r)
            g = self.pil_from_array(g)
            b = self.pil_from_array(b)

            return Image.merge("RGB", (r, g, b))
        elif self.profile == "PIL":
            return image.convert("RGB")
        else:
            cms_profile_cmyk = ImageCms.getOpenProfile(
                COLOR_PROFILES_DIR + color_profiles[self.profile]
            )
            cms_profile_srgb = ImageCms.createProfile("sRGB")
            cms_xform = ImageCms.buildTransformFromOpenProfiles(
                cms_profile_cmyk, cms_profile_srgb, "CMYK", "RGB",
                renderingIntent=ImageCms.Intent.RELATIVE_COLORIMETRIC,
                flags=(ImageCms.FLAGS["BLACKPOINTCOMPENSATION"] |
                       ImageCms.FLAGS["HIGHRESPRECALC"]),
            )
            return ImageCms.applyTransform(image, cms_xform)

    def invoke(self, context: InvocationContext) -> CMYKSeparationOutput:
        image_size = self.width, self.height
        c_image_mix = Image.new("L", image_size, color=round(2.55*self.c_value))
        m_image_mix = Image.new("L", image_size, color=round(2.55*self.m_value))
        y_image_mix = Image.new("L", image_size, color=round(2.55*self.y_value))
        k_image_mix = Image.new("L", image_size, color=round(2.55*self.k_value))
        c_image_a = Image.new("L", image_size, color=round(2.55*(math.floor(self.c_value*self.c_split))))
        m_image_a = Image.new("L", image_size, color=round(2.55*(math.floor(self.m_value*self.m_split))))
        y_image_a = Image.new("L", image_size, color=round(2.55*(math.floor(self.y_value*self.y_split))))
        k_image_a = Image.new("L", image_size, color=round(2.55*(math.floor(self.k_value*self.k_split))))
        c_image_b = Image.new("L", image_size, color=round(2.55*(math.ceil(self.c_value*(1.-self.c_split)))))
        m_image_b = Image.new("L", image_size, color=round(2.55*(math.ceil(self.m_value*(1.-self.m_split)))))
        y_image_b = Image.new("L", image_size, color=round(2.55*(math.ceil(self.y_value*(1.-self.y_split)))))
        k_image_b = Image.new("L", image_size, color=round(2.55*(math.ceil(self.k_value*(1.-self.k_split)))))

        cmyk_mix = Image.merge("CMYK", (c_image_mix, m_image_mix, y_image_mix, k_image_mix))
        cmyk_a = Image.merge("CMYK", (c_image_a, m_image_a, y_image_a, k_image_a))
        cmyk_b = Image.merge("CMYK", (c_image_b, m_image_b, y_image_b, k_image_b))

        image_mix = self.convert_cmyk_to_rgb(cmyk_mix)
        image_a = self.convert_cmyk_to_rgb(cmyk_a)
        image_b = self.convert_cmyk_to_rgb(cmyk_b)

        array_a = numpy.array(image_a)
        array_b = numpy.array(image_b)

        rgb_red_a   = int(array_a[0,0,0])
        rgb_green_a = int(array_a[0,0,1])
        rgb_blue_a  = int(array_a[0,0,2])
        rgb_red_b   = int(array_b[0,0,0])
        rgb_green_b = int(array_b[0,0,1])
        rgb_blue_b  = int(array_b[0,0,2])

        image_mix = Image.fromarray(numpy.array(image_mix))
        image_a = Image.fromarray(array_a)
        image_b = Image.fromarray(array_b)

        image_dto_mix = context.services.images.create(
            image=image_mix,
            image_origin=ResourceOrigin.INTERNAL,
            image_category=ImageCategory.GENERAL,
            node_id=self.id,
            session_id=context.graph_execution_state_id,
            is_intermediate=self.is_intermediate,
            metadata=self.metadata,
            workflow=context.workflow,
        )
        image_dto_a = context.services.images.create(
            image=image_a,
            image_origin=ResourceOrigin.INTERNAL,
            image_category=ImageCategory.GENERAL,
            node_id=self.id,
            session_id=context.graph_execution_state_id,
            is_intermediate=self.is_intermediate,
            metadata=self.metadata,
            workflow=context.workflow,
        )
        image_dto_b = context.services.images.create(
            image=image_b,
            image_origin=ResourceOrigin.INTERNAL,
            image_category=ImageCategory.GENERAL,
            node_id=self.id,
            session_id=context.graph_execution_state_id,
            is_intermediate=self.is_intermediate,
            metadata=self.metadata,
            workflow=context.workflow,
        )

        return CMYKSeparationOutput(
            color_image=ImageField(image_name=image_dto_mix.image_name),
            width=self.width,
            height=self.height,
            part_a=ImageField(image_name=image_dto_a.image_name),
            rgb_red_a=rgb_red_a,
            rgb_green_a=rgb_green_a,
            rgb_blue_a=rgb_blue_a,
            rgb_red_b=rgb_red_b,
            rgb_green_b=rgb_green_b,
            rgb_blue_b=rgb_blue_b,
            part_b=ImageField(image_name=image_dto_b.image_name),
        )


# @invocation("cmyk_add", title="CMYK Image Add", tags=["image", "cmyk", "add", "blend"], category="image")
# class CMYKImageAddInvocation(BaseInvocation):

