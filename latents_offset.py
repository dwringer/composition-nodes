
import torch


from invokeai.app.invocations.baseinvocation import (
    BaseInvocation,
    FieldDescriptions,
    Input,
    InputField,
    InvocationContext,
    invocation,
)
from invokeai.backend.util.devices import choose_torch_device
from invokeai.app.invocations.primitives import (
    LatentsField,
    LatentsOutput,
    build_latents_output,
)


@invocation(
    "offset_latents",
    title="Offset Latents",
    tags=["latents", "offset"],
    category="latents")
class OffsetLatentsInvocation(BaseInvocation):
    latents: LatentsField = InputField(
        description=FieldDescriptions.latents,
        input=Input.Connection,
    )
    x_offset: float = InputField(default=0.5, description="Approx percentage to offset (H)")
    y_offset: float = InputField(default=0.5, description="Approx percentage to offset (V)")

    def invoke(self, context: InvocationContext) -> LatentsOutput:
        latents = context.services.latents.get(self.latents.latents_name)
        device = choose_torch_device()
        x_offset = int(self.x_offset * latents.size(dim=2))
        y_offset = int(self.y_offset * latents.size(dim=3))
        latents_out = torch.roll(
            latents.to(device),
            shifts=(x_offset, y_offset),
            dims=(2, 3)
        )
        latents_out = latents_out.to("cpu")
        torch.cuda.empty_cache()
        name = f"{context.graph_execution_state_id}__{self.id}"
        context.services.latents.save(name, latents_out)
        return build_latents_output(latents_name=name, latents=latents_out, seed=self.latents.seed)
