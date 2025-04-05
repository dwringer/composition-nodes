import torch

from invokeai.invocation_api import (
    BaseInvocation,
    FieldDescriptions,
    Input,
    InputField,
    InvocationContext,
    LatentsField,
    LatentsOutput,
    choose_torch_device,
    invocation,
)


@invocation(
    "offset_latents",
    title="Offset Latents",
    tags=["latents", "offset"],
    category="latents",
    version="1.2.0",
)
class OffsetLatentsInvocation(BaseInvocation):
    """Offsets a latents tensor by a given percentage of height/width.

This takes a Latents input as well as two numbers (between 0 and 1), which are used to offset the latents in the vertical and/or horizontal directions. 0.5/0.5 would offset the image 50% in both directions such that the corners will wrap around and become the center of the image.
"""

    latents: LatentsField = InputField(
        description=FieldDescriptions.latents,
        input=Input.Connection,
    )
    x_offset: float = InputField(default=0.5, description="Approx percentage to offset (H)")
    y_offset: float = InputField(default=0.5, description="Approx percentage to offset (V)")

    def invoke(self, context: InvocationContext) -> LatentsOutput:
        latents = context.tensors.load(self.latents.latents_name)
        device = choose_torch_device()
        x_offset = int(self.x_offset * latents.size(dim=2))
        y_offset = int(self.y_offset * latents.size(dim=3))
        latents_out = torch.roll(latents.to(device), shifts=(x_offset, y_offset), dims=(2, 3))
        latents_out = latents_out.to("cpu")
        torch.cuda.empty_cache()
        name = context.tensors.save(tensor=latents_out)

        return LatentsOutput.build(latents_name=name, latents=latents_out)
