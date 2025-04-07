from typing import Optional

import numpy as np
import scipy.ndimage as gaussian_filter
import torch
import torchvision.transforms as T
from PIL import Image
from torchvision.transforms.functional import resize as tv_resize

from invokeai.app.invocations.baseinvocation import BaseInvocation, invocation
from invokeai.app.invocations.fields import FieldDescriptions, ImageField, Input, InputField, LatentsField
from invokeai.app.invocations.primitives import LatentsOutput
from invokeai.app.services.shared.invocation_context import InvocationContext
from invokeai.backend.stable_diffusion.diffusers_pipeline import image_resized_to_grid_as_tensor
from invokeai.backend.util.devices import TorchDevice


@invocation(
    "frequency_match_latents",
    title="Frequency Spectrum Match Latents",
    tags=["latents", "noise", "frequency", "mask", "blend"],
    category="latents",
    version="1.0.0",
)
class FrequencySpectrumMatchLatentsInvocation(BaseInvocation):
    """
    Generates latent noise with the frequency spectrum of target latents, masked by a provided mask image.
    Takes both target latents and white noise latents as inputs.
    """

    target_latents: LatentsField = InputField(description="Target latents to match frequency spectrum", input=Input.Connection)
    white_noise_latents: LatentsField = InputField(description="White noise latents for phase information", input=Input.Connection)
    mask: Optional[ImageField] = InputField(default=None, description="Mask for blending (optional)")
    blur_sigma: float = InputField(ge=0, default=0, description="Amount of Gaussian blur to apply to the mask")
    
    def prep_mask_tensor(self, mask_image: Image.Image) -> torch.Tensor:
        """Convert mask image to tensor and resize to match latents dimensions"""
        if mask_image.mode != "L":
            mask_image = mask_image.convert("L")
        mask_tensor = image_resized_to_grid_as_tensor(mask_image, normalize=False)
        if mask_tensor.dim() == 3:
            mask_tensor = mask_tensor.unsqueeze(0)
        return mask_tensor
    
    def match_frequency_spectrum_masked(self, target_latents, white_noise_latents, mask_tensor=None, device=None):
        """
        Generates latent noise with the frequency spectrum of target latents,
        optionally masked by a provided mask tensor.
        """
        # Process each channel separately
        batch_size, channels, height, width = target_latents.shape
        result = torch.zeros_like(target_latents, device=device)
        
        # Convert tensors to numpy for FFT processing
        target_np = target_latents.float().detach().cpu().numpy()
        white_noise_np = white_noise_latents.float().detach().cpu().numpy()
        
        for b in range(batch_size):
            for c in range(channels):
                # Compute FFT of target latents
                target_fft = np.fft.fft2(target_np[b, c])
                target_amplitude = np.abs(target_fft)
                
                # Compute FFT of white noise latents
                white_noise_fft = np.fft.fft2(white_noise_np[b, c])
                white_noise_phase = np.angle(white_noise_fft)
                
                # Combine amplitude from target and phase from white noise
                matched_fft = target_amplitude * np.exp(1j * white_noise_phase)
                
                # Convert back to spatial domain
                matched_noise = np.fft.ifft2(matched_fft).real
                
                # Convert back to torch tensor
                result[b, c] = torch.from_numpy(matched_noise).to(device)
        
        # Apply mask and blend if provided
        if mask_tensor is not None:
            # Expand mask to match latents dimensions
            expanded_mask = mask_tensor.expand(target_latents.shape)
            if target_latents.dtype == torch.float16:
                expanded_mask = expanded_mask.half()
            
            # Blend based on mask
            result = expanded_mask * white_noise_latents + (1 - expanded_mask) * result
        
        return result

    def invoke(self, context: InvocationContext) -> LatentsOutput:
        # Load input latents
        target_latents = context.tensors.load(self.target_latents.latents_name)
        white_noise_latents = context.tensors.load(self.white_noise_latents.latents_name)
        
        # Ensure white noise has the same shape as target
        if white_noise_latents.shape != target_latents.shape:
            raise ValueError("Target latents and white noise latents must have the same shape")
        
        device = TorchDevice.choose_torch_device()
        target_latents = target_latents.to(device)
        white_noise_latents = white_noise_latents.to(device)
        
        # Process mask if provided
        mask_tensor = None
        if self.mask is not None:
            mask_image = context.images.get_pil(self.mask.image_name)
            mask_tensor = self.prep_mask_tensor(mask_image)
            mask_tensor = tv_resize(mask_tensor, target_latents.shape[-2:], T.InterpolationMode.BILINEAR, antialias=False)
            mask_tensor = mask_tensor.to(device)
            
            # Apply Gaussian blur if specified
            if self.blur_sigma > 0:
                mask_numpy = mask_tensor.cpu().numpy().squeeze(0).squeeze(0)  # Convert to numpy and remove unnecessary dimensions
                blurred_mask_numpy = gaussian_filter(mask_numpy, sigma=self.blur_sigma)
                mask_tensor = torch.from_numpy(blurred_mask_numpy).unsqueeze(0).unsqueeze(0).to(mask_tensor.device) # Convert back to tensor with original dimensions and device
        
        # Apply frequency spectrum matching
        result_latents = self.match_frequency_spectrum_masked(
            target_latents, 
            white_noise_latents, 
            mask_tensor,
            device
        )
        
        # Move result to CPU to free GPU memory
        result_latents = result_latents.to("cpu")
        torch.cuda.empty_cache()
        
        # Save the result
        name = context.tensors.save(tensor=result_latents)
        
        return LatentsOutput.build(latents_name=name, latents=result_latents)


@invocation(
    "frequency_match_phase_blend_latents",
    title="Freq. Match & Phase Blend Latents",
    tags=["latents", "noise", "frequency", "mask", "blend", "phase"],
    category="latents",
    version="1.0.0",
)
class FrequencyMatchPhaseBlendLatentsInvocation(BaseInvocation):
    """
    Generates latent noise with the frequency spectrum of target latents, masked by a provided mask image.
    Takes both target latents and white noise latents as inputs.
    """

    target_latents: LatentsField = InputField(description="Target latents to match frequency spectrum", input=Input.Connection)
    phase_latents_a: LatentsField = InputField(description="White noise latents for phase information (A)", input=Input.Connection)
    phase_latents_b: LatentsField = InputField(description="White noise latents for phase information (B)", input=Input.Connection)
    phase_blend_alpha: float = InputField(ge=0, le=1, default=0.0, description="Blend ratio for the phases")
    mask: Optional[ImageField] = InputField(default=None, description="Mask for blending (optional)")
    blur_sigma: float = InputField(ge=0, default=0, description="Amount of Gaussian blur to apply to the mask")
    
    def prep_mask_tensor(self, mask_image: Image.Image) -> torch.Tensor:
        """Convert mask image to tensor and resize to match latents dimensions"""
        if mask_image.mode != "L":
            mask_image = mask_image.convert("L")
        mask_tensor = image_resized_to_grid_as_tensor(mask_image, normalize=False)
        if mask_tensor.dim() == 3:
            mask_tensor = mask_tensor.unsqueeze(0)
        return mask_tensor
    
    def match_frequency_spectrum_masked(self, target_latents, phase_latents_a, phase_latents_b, mask_tensor=None, device=None):
        """
        Generates latent noise with the frequency spectrum of target latents,
        optionally masked by a provided mask tensor.
        """
        # Process each channel separately
        batch_size, channels, height, width = target_latents.shape
        result = torch.zeros_like(target_latents, device=device)
        
        # Convert tensors to numpy for FFT processing
        target_np = target_latents.float().detach().cpu().numpy()
        phase_a_np = phase_latents_a.float().detach().cpu().numpy()
        phase_b_np = phase_latents_b.float().detach().cpu().numpy()
        
        for b in range(batch_size):
            for c in range(channels):
                # Compute FFT of target latents
                target_fft = np.fft.fft2(target_np[b, c])
                target_amplitude = np.abs(target_fft)
                
                # Compute FFT of white noise latents
                phase_a_fft = np.fft.fft2(phase_a_np[b, c])
                phase_a_phases = np.angle(phase_a_fft)
                phase_b_fft = np.fft.fft2(phase_b_np[b, c])
                phase_b_phases = np.angle(phase_b_fft)
                phase_difference = (phase_b_phases - phase_a_phases + np.pi) % (2 * np.pi) - np.pi
                phase_blend = phase_a_phases + phase_difference * self.phase_blend_alpha
                
                # Combine amplitude from target and phase from white noise
                matched_fft = target_amplitude * np.exp(1j * phase_blend)
                
                # Convert back to spatial domain
                matched_noise = np.fft.ifft2(matched_fft).real
                
                # Convert back to torch tensor
                result[b, c] = torch.from_numpy(matched_noise).to(device)
        
        # Apply mask and blend if provided
        if mask_tensor is not None:
            # Expand mask to match latents dimensions
            expanded_mask = mask_tensor.expand(target_latents.shape)
            if target_latents.dtype == torch.float16:
                expanded_mask = expanded_mask.half()
            
            # Blend based on mask
            result = expanded_mask * phase_latents_a + (1 - expanded_mask) * result
        
        return result

    def invoke(self, context: InvocationContext) -> LatentsOutput:
        # Load input latents
        target_latents = context.tensors.load(self.target_latents.latents_name)
        phase_latents_a = context.tensors.load(self.phase_latents_a.latents_name)
        phase_latents_b = context.tensors.load(self.phase_latents_b.latents_name)
        
        # Ensure phase input latents have the same shape as target
        if (phase_latents_a.shape != target_latents.shape) or  \
           (phase_latents_b.shape != target_latents.shape):
            raise ValueError("Target latents and phase latents must have the same shape")
        
        device = TorchDevice.choose_torch_device()
        target_latents = target_latents.to(device)
        phase_latents_a = phase_latents_a.to(device)
        phase_latents_b = phase_latents_b.to(device)
        
        # Process mask if provided
        mask_tensor = None
        if self.mask is not None:
            mask_image = context.images.get_pil(self.mask.image_name)
            mask_tensor = self.prep_mask_tensor(mask_image)
            mask_tensor = tv_resize(mask_tensor, target_latents.shape[-2:], T.InterpolationMode.BILINEAR, antialias=False)
            mask_tensor = mask_tensor.to(device)
            
            # Apply Gaussian blur if specified
            if self.blur_sigma > 0:
                mask_numpy = mask_tensor.cpu().numpy().squeeze(0).squeeze(0)  # Convert to numpy and remove unnecessary dimensions
                blurred_mask_numpy = gaussian_filter(mask_numpy, sigma=self.blur_sigma)
                mask_tensor = torch.from_numpy(blurred_mask_numpy).unsqueeze(0).unsqueeze(0).to(mask_tensor.device) # Convert back to tensor with original dimensions and device
        
        # Apply frequency spectrum matching
        result_latents = self.match_frequency_spectrum_masked(
            target_latents, 
            phase_latents_a, 
            phase_latents_b, 
            mask_tensor,
            device
        )
        
        # Move result to CPU to free GPU memory
        result_latents = result_latents.to("cpu")
        torch.cuda.empty_cache()
        
        # Save the result
        name = context.tensors.save(tensor=result_latents)
        
        return LatentsOutput.build(latents_name=name, latents=result_latents)


@invocation(
    "frequency_blend_match_phase_blend_latents",
    title="Freq. Blend Match & Phase Blend",
    tags=["latents", "noise", "frequency", "mask", "blend", "phase"],
    category="latents",
    version="1.0.0",
)
class FrequencyBlendMatchPhaseBlendLatentsInvocation(BaseInvocation):
    """
    Generates latent noise with the frequency spectrum of target latents, masked by a provided mask image.
    Takes both target latents and white noise latents as inputs.
    """

    target_latents_a: LatentsField = InputField(description="Target latents to match frequency spectrum (A)", input=Input.Connection)
    target_latents_b: LatentsField = InputField(description="Target latents to match frequency spectrum (B)", input=Input.Connection)
    frequency_blend_alpha: float = InputField(ge=0, le=1, default=0.0, description="Blend ratio for the frequency spectra")
    phase_latents_a: LatentsField = InputField(description="White noise latents for phase information (A)", input=Input.Connection)
    phase_latents_b: LatentsField = InputField(description="White noise latents for phase information (B)", input=Input.Connection)
    phase_blend_alpha: float = InputField(ge=0, le=1, default=0.0, description="Blend ratio for the phases")
    mask: Optional[ImageField] = InputField(default=None, description="Mask for blending (optional)")
    blur_sigma: float = InputField(ge=0, default=0, description="Amount of Gaussian blur to apply to the mask")
    
    def prep_mask_tensor(self, mask_image: Image.Image) -> torch.Tensor:
        """Convert mask image to tensor and resize to match latents dimensions"""
        if mask_image.mode != "L":
            mask_image = mask_image.convert("L")
        mask_tensor = image_resized_to_grid_as_tensor(mask_image, normalize=False)
        if mask_tensor.dim() == 3:
            mask_tensor = mask_tensor.unsqueeze(0)
        return mask_tensor
    
    def match_frequency_spectrum_masked(self, target_latents_a, target_latents_b, phase_latents_a, phase_latents_b, mask_tensor=None, device=None):
        """
        Generates latent noise with the frequency spectrum of target latents,
        optionally masked by a provided mask tensor.
        """
        # Process each channel separately
        batch_size, channels, height, width = target_latents_a.shape
        result = torch.zeros_like(target_latents_a, device=device)
        
        # Convert tensors to numpy for FFT processing
        target_a_np = target_latents_a.float().detach().cpu().numpy()
        target_b_np = target_latents_b.float().detach().cpu().numpy()
        phase_a_np = phase_latents_a.float().detach().cpu().numpy()
        phase_b_np = phase_latents_b.float().detach().cpu().numpy()
        
        for b in range(batch_size):
            for c in range(channels):
                # Compute FFT of target latents
                target_a_fft = np.fft.fft2(target_a_np[b, c])
                target_a_amplitude = np.abs(target_a_fft)
                target_b_fft = np.fft.fft2(target_b_np[b, c])
                target_b_amplitude = np.abs(target_b_fft)
                amplitude_blend = self.frequency_blend_alpha * target_b_amplitude + (1-self.frequency_blend_alpha) * target_a_amplitude
                
                # Compute FFT of white noise latents
                phase_a_fft = np.fft.fft2(phase_a_np[b, c])
                phase_a_phases = np.angle(phase_a_fft)
                phase_b_fft = np.fft.fft2(phase_b_np[b, c])
                phase_b_phases = np.angle(phase_b_fft)
                phase_difference = (phase_b_phases - phase_a_phases + np.pi) % (2 * np.pi) - np.pi
                phase_blend = phase_a_phases + phase_difference * self.phase_blend_alpha
                
                # Combine amplitude from target and phase from white noise
                matched_fft = amplitude_blend * np.exp(1j * phase_blend)
                
                # Convert back to spatial domain
                matched_noise = np.fft.ifft2(matched_fft).real
                
                # Convert back to torch tensor
                result[b, c] = torch.from_numpy(matched_noise).to(device)
        
        # Apply mask and blend if provided
        if mask_tensor is not None:
            # Expand mask to match latents dimensions
            expanded_mask = mask_tensor.expand(target_latents_a.shape)
            if target_latents_a.dtype == torch.float16:
                expanded_mask = expanded_mask.half()
            
            # Blend based on mask
            result = expanded_mask * phase_latents_a + (1 - expanded_mask) * result
        
        return result

    def invoke(self, context: InvocationContext) -> LatentsOutput:
        # Load input latents
        target_latents_a = context.tensors.load(self.target_latents_a.latents_name)
        target_latents_b = context.tensors.load(self.target_latents_b.latents_name)
        phase_latents_a = context.tensors.load(self.phase_latents_a.latents_name)
        phase_latents_b = context.tensors.load(self.phase_latents_b.latents_name)
        
        # Ensure phase input latents have the same shape as target
        if (phase_latents_a.shape != target_latents_a.shape) or  \
           (phase_latents_b.shape != target_latents_a.shape) or  \
           (target_latents_a.shape != target_latents_b.shape):
            raise ValueError("Target latents and phase latents must have the same shape")
        
        device = TorchDevice.choose_torch_device()
        target_latents_a = target_latents_a.to(device)
        target_latents_b = target_latents_b.to(device)
        phase_latents_a = phase_latents_a.to(device)
        phase_latents_b = phase_latents_b.to(device)
        
        # Process mask if provided
        mask_tensor = None
        if self.mask is not None:
            mask_image = context.images.get_pil(self.mask.image_name)
            mask_tensor = self.prep_mask_tensor(mask_image)
            mask_tensor = tv_resize(mask_tensor, target_latents_a.shape[-2:], T.InterpolationMode.BILINEAR, antialias=False)
            mask_tensor = mask_tensor.to(device)
            
            # Apply Gaussian blur if specified
            if self.blur_sigma > 0:
                mask_numpy = mask_tensor.cpu().numpy().squeeze(0).squeeze(0)  # Convert to numpy and remove unnecessary dimensions
                blurred_mask_numpy = gaussian_filter(mask_numpy, sigma=self.blur_sigma)
                mask_tensor = torch.from_numpy(blurred_mask_numpy).unsqueeze(0).unsqueeze(0).to(mask_tensor.device) # Convert back to tensor with original dimensions and device
        
        # Apply frequency spectrum matching
        result_latents = self.match_frequency_spectrum_masked(
            target_latents_a,
            target_latents_b,
            phase_latents_a, 
            phase_latents_b, 
            mask_tensor,
            device
        )
        
        # Move result to CPU to free GPU memory
        result_latents = result_latents.to("cpu")
        torch.cuda.empty_cache()
        
        # Save the result
        name = context.tensors.save(tensor=result_latents)
        
        return LatentsOutput.build(latents_name=name, latents=result_latents)
