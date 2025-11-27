import os
import json
from pathlib import Path
from datetime import datetime
from PIL import Image
from typing import Dict, List
from .utils import get_date_folder, sanitize_filename, get_timestamp

class ImageStorage:
    def __init__(self, base_output_dir: str = "outputs"):
        """
        Initialize storage manager
        
        Args:
            base_output_dir: Base directory for outputs
        """
        self.base_output_dir = Path(base_output_dir)
        self.base_output_dir.mkdir(exist_ok=True)
        
    def _get_output_folder(self) -> Path:
        """Create and return date-based output folder"""
        folder = self.base_output_dir / get_date_folder()
        folder.mkdir(exist_ok=True)
        return folder
    
    def save_images(
        self,
        images: List[Image.Image],
        metadata: Dict,
        prompt: str,
        format: str = "PNG"
    ) -> List[str]:
        """
        Save generated images with metadata
        
        Args:
            images: List of PIL images
            metadata: Generation metadata
            prompt: Original prompt text
            format: Image format (PNG or JPEG)
            
        Returns:
            List of saved file paths
        """
        output_folder = self._get_output_folder()
        timestamp = get_timestamp()
        prompt_slug = sanitize_filename(prompt, max_length=30)
        
        saved_paths = []
        
        for idx, image in enumerate(images):
            # Generate filename
            filename = f"{timestamp}_{prompt_slug}_{idx+1}.{format.lower()}"
            filepath = output_folder / filename
            
            # Save image
            if format.upper() == "JPEG":
                image = image.convert("RGB")  # JPEG doesn't support transparency
                image.save(filepath, format="JPEG", quality=95)
            else:
                image.save(filepath, format="PNG")
            
            saved_paths.append(str(filepath))
            
            # Save individual metadata
            metadata_copy = metadata.copy()
            metadata_copy["image_index"] = idx + 1
            metadata_copy["filename"] = filename
            
            metadata_path = filepath.with_suffix('.json')
            self._save_metadata(metadata_copy, metadata_path)
        
        # Save batch metadata
        batch_metadata_path = output_folder / f"{timestamp}_batch_metadata.json"
        batch_metadata = metadata.copy()
        batch_metadata["num_images"] = len(images)
        batch_metadata["saved_files"] = [Path(p).name for p in saved_paths]
        self._save_metadata(batch_metadata, batch_metadata_path)
        
        return saved_paths
    
    def _save_metadata(self, metadata: Dict, filepath: Path):
        """Save metadata to JSON file"""
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    def create_metadata(
        self,
        prompt: str,
        negative_prompt: str,
        num_inference_steps: int,
        guidance_scale: float,
        seed: int,
        width: int,
        height: int,
        model_id: str,
        style: str = "none",
        device: str = "unknown"
    ) -> Dict:
        """
        Create metadata dictionary
        
        Returns:
            Metadata dictionary
        """
        return {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "model": model_id,
            "parameters": {
                "num_inference_steps": num_inference_steps,
                "guidance_scale": guidance_scale,
                "seed": seed,
                "width": width,
                "height": height,
                "style_preset": style
            },
            "device": device,
            "timestamp": datetime.now().isoformat(),
            "generated_with": "Stable Diffusion Text-to-Image"
        }
