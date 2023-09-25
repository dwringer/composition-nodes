import numpy as np
import torch
import torchvision.transforms as T
from torchvision.transforms.functional import resize as tv_resize

from invokeai.app.invocations.primitives import (
    ImageField,
    LatentsField,
    LatentsOutput,
    build_latents_output,
)

from invokeai.backend.stable_diffusion.diffusers_pipeline import (
    image_resized_to_grid_as_tensor,
)
from invokeai.backend.util.devices import choose_torch_device
from invokeai.app.invocations.baseinvocation import (
    BaseInvocation,
    FieldDescriptions,
    Input,
    InputField,
    InvocationContext,
    invocation,
)


@invocation("lmblend", title="Blend Latents/Noise (Masked)", tags=["latents", "noise", "blend"], category="latents", version="1.0.1")
class MaskedBlendLatentsInvocation(BaseInvocation):
    """Blend two latents using a given alpha and mask. Latents must have same size."""

    latents_a: LatentsField = InputField(
        description=FieldDescriptions.latents,
        input=Input.Connection,
    )
    latents_b: LatentsField = InputField(
        description=FieldDescriptions.latents,
        input=Input.Connection,
    )
    mask: ImageField = InputField(description="Mask for blending in latents B")
    alpha: float = InputField(default=0.5, description=FieldDescriptions.blend_alpha)

    def prep_mask_tensor(self, mask_image):
        if mask_image.mode != "L":
            mask_image = mask_image.convert("L")
        mask_tensor = image_resized_to_grid_as_tensor(mask_image, normalize=False)
        if mask_tensor.dim() == 3:
            mask_tensor = mask_tensor.unsqueeze(0)
        return mask_tensor

    def replace_tensor_from_masked_tensor(self, tensor, other_tensor, mask_tensor):
        output = tensor.clone()
        mask_tensor = mask_tensor.expand(output.shape)
        if output.dtype != torch.float16:
            output = torch.add(output, mask_tensor * torch.sub(other_tensor, tensor))
        else:
            output = torch.add(output, mask_tensor.half() * torch.sub(other_tensor, tensor))
        return output

    def invoke(self, context: InvocationContext) -> LatentsOutput:
        latents_a = context.services.latents.get(self.latents_a.latents_name)
        latents_b = context.services.latents.get(self.latents_b.latents_name)
        mask_tensor = self.prep_mask_tensor(
            context.services.images.get_pil_image(self.mask.image_name)
        )

        mask_tensor = tv_resize(mask_tensor, latents_a.shape[-2:], T.InterpolationMode.BILINEAR, antialias=False)

        latents_b = self.replace_tensor_from_masked_tensor(latents_b, latents_a, mask_tensor)
       
        if latents_a.shape != latents_b.shape:
            raise "Latents to blend must be the same size."

        # TODO:
        device = choose_torch_device()

        def slerp(t, v0, v1, DOT_THRESHOLD=0.9995):
            """
            Spherical linear interpolation
            Args:
                t (float/np.ndarray): Float value between 0.0 and 1.0
                v0 (np.ndarray): Starting vector
                v1 (np.ndarray): Final vector
                DOT_THRESHOLD (float): Threshold for considering the two vectors as
                                    colineal. Not recommended to alter this.
            Returns:
                v2 (np.ndarray): Interpolation vector between v0 and v1
            """
            inputs_are_torch = False
            if not isinstance(v0, np.ndarray):
                inputs_are_torch = True
                v0 = v0.detach().cpu().numpy()
            if not isinstance(v1, np.ndarray):
                inputs_are_torch = True
                v1 = v1.detach().cpu().numpy()

            dot = np.sum(v0 * v1 / (np.linalg.norm(v0) * np.linalg.norm(v1)))
            if np.abs(dot) > DOT_THRESHOLD:
                v2 = (1 - t) * v0 + t * v1
            else:
                theta_0 = np.arccos(dot)
                sin_theta_0 = np.sin(theta_0)
                theta_t = theta_0 * t
                sin_theta_t = np.sin(theta_t)
                s0 = np.sin(theta_0 - theta_t) / sin_theta_0
                s1 = sin_theta_t / sin_theta_0
                v2 = s0 * v0 + s1 * v1

            if inputs_are_torch:
                v2 = torch.from_numpy(v2).to(device)

            return v2

        # blend
        blended_latents = slerp(self.alpha, latents_a, latents_b)

        # https://discuss.huggingface.co/t/memory-usage-by-later-pipeline-stages/23699
        blended_latents = blended_latents.to("cpu")
        torch.cuda.empty_cache()

        name = f"{context.graph_execution_state_id}__{self.id}"
        # context.services.latents.set(name, resized_latents)
        context.services.latents.save(name, blended_latents)
        return build_latents_output(latents_name=name, latents=blended_latents)    
