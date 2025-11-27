import torch
from diffusers import StableDiffusionPipeline
from typing import List, Tuple, Optional
import time

class ImageGenerator:
    def __init__(self, model_id: str = "runwayml/stable-diffusion-v1-5"):
        """
        Initialize the image generator with MPS-optimized settings
        
        Args:
            model_id: HuggingFace model identifier
        """
        self.model_id = model_id
        self.device = self._get_device()
        self.pipe = None
        
    def _get_device(self) -> str:
        """Detect and return the best available device"""
        if torch.backends.mps.is_available():
            return "mps"
        elif torch.cuda.is_available():
            return "cuda"
        else:
            return "cpu"
    
    def load_model(self):
        """Load the Stable Diffusion model with MPS-compatible settings"""
        print(f"Loading model: {self.model_id}")
        print(f"Target device: {self.device}")
        
        # Load AbsoluteReality
        self.pipe = StableDiffusionPipeline.from_pretrained(
            self.model_id,
            torch_dtype=torch.float32,  # MPS needs float32
            use_safetensors=True
        )
        
        # CRITICAL: Disable safety checker (causes CPU fallback on MPS)
        self.pipe.safety_checker = None
        self.pipe.feature_extractor = None
        
        # Move ALL components to MPS explicitly
        print("Moving components to MPS...")
        self.pipe = self.pipe.to(self.device)
        self.pipe.unet = self.pipe.unet.to(self.device)
        self.pipe.vae = self.pipe.vae.to(self.device)
        self.pipe.text_encoder = self.pipe.text_encoder.to(self.device)
        
        # Enable attention slicing (critical for 16GB RAM)
        self.pipe.enable_attention_slicing()
        
        # Verify everything is on MPS
        print("\n" + "="*50)
        print("Device Verification:")
        print(f"UNet: {next(self.pipe.unet.parameters()).device}")
        print(f"VAE: {next(self.pipe.vae.parameters()).device}")
        print(f"Text Encoder: {next(self.pipe.text_encoder.parameters()).device}")
        print("="*50 + "\n")
        
        # Check if any component is still on CPU
        unet_device = str(next(self.pipe.unet.parameters()).device)
        if "cpu" in unet_device:
            raise RuntimeError("⚠️ Model is on CPU! MPS not working properly")
        
        print("✅ AbsoluteReality loaded successfully on MPS")

        
    def generate_images(
        self,
        prompt: str,
        negative_prompt: str = "",
        num_images: int = 1,
        num_inference_steps: int = 30,
        guidance_scale: float = 7.5,
        width: int = 512,
        height: int = 512,
        seed: Optional[int] = None
    ) -> Tuple[List, int]:
        """
        Generate images from text prompt
        
        Args:
            prompt: Text description of desired image
            negative_prompt: What to avoid in the image
            num_images: Number of images to generate
            num_inference_steps: Number of denoising steps (10-50)
            guidance_scale: How closely to follow prompt (1-20)
            width: Image width in pixels
            height: Image height in pixels
            seed: Random seed for reproducibility
            
        Returns:
            Tuple of (list of PIL images, used seed)
        """
        if self.pipe is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        # Set seed for reproducibility
        if seed is None:
            seed = int(time.time())
        
        generator = torch.Generator(device=self.device).manual_seed(seed)
        
        # Generate images
        print(f"Generating {num_images} image(s) with seed {seed}...")
        
        output = self.pipe(
            prompt=prompt,
            negative_prompt=negative_prompt if negative_prompt else None,
            num_images_per_prompt=num_images,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            width=width,
            height=height,
            generator=generator
        )
        
        images = output.images
        print(f"Successfully generated {len(images)} image(s)")
        
        return images, seed
    
    def get_device_info(self) -> dict:
        """Return device and memory information"""
        info = {
            "device": self.device,
            "model_loaded": self.pipe is not None
        }
        
        if self.device == "mps":
            info["backend"] = "Metal Performance Shaders (Apple Silicon)"
        elif self.device == "cuda":
            info["backend"] = f"CUDA {torch.version.cuda}"
            info["gpu_name"] = torch.cuda.get_device_name(0)
        else:
            info["backend"] = "CPU"
            
        return info
