import re
from datetime import datetime
from typing import Dict

# Style templates for prompt enhancement
STYLE_PRESETS = {
    "none": {
        "suffix": "",
        "negative": ""
    },
    "photorealistic": {
        "suffix": ", highly detailed, sharp focus, professional photography, 4K, realistic lighting",
        "negative": "cartoon, painting, illustration, drawing, art, sketch, anime"
    },
    "artistic": {
        "suffix": ", beautiful artwork, oil painting, masterpiece, trending on artstation, by greg rutkowski",
        "negative": "photo, photograph, realistic, realism"
    },
    "anime": {
        "suffix": ", anime style, manga, detailed, vibrant colors, studio quality",
        "negative": "realistic, photograph, 3d render"
    },
    "cinematic": {
        "suffix": ", cinematic lighting, dramatic, epic composition, movie still, 8K, volumetric lighting",
        "negative": "amateur, low quality, simple"
    },
    "fantasy": {
        "suffix": ", fantasy art, magical, ethereal, mystical, detailed, concept art",
        "negative": "modern, urban, realistic"
    },
    "sci-fi": {
        "suffix": ", science fiction, futuristic, cyberpunk, detailed, concept art, neon lights",
        "negative": "medieval, fantasy, historical"
    },
    "portrait": {
        "suffix": ", professional portrait, studio lighting, sharp focus, detailed face, 85mm lens, f/1.8",
        "negative": "blurry, distorted, deformed, bad anatomy, extra limbs"
    },
    "landscape": {
        "suffix": ", beautiful landscape, wide angle, golden hour, highly detailed, nature photography",
        "negative": "people, portrait, urban, indoor"
    }
}

# Default negative prompts
DEFAULT_NEGATIVE_PROMPT = "lowres, bad quality, blurry, watermark, text, signature, username, cropped, worst quality, jpeg artifacts"

def apply_style_preset(prompt: str, style: str, negative_prompt: str = "") -> tuple:
    """
    Apply style preset to prompt and negative prompt
    
    Args:
        prompt: Original text prompt
        style: Style preset name
        negative_prompt: Original negative prompt
        
    Returns:
        Tuple of (enhanced_prompt, enhanced_negative_prompt)
    """
    preset = STYLE_PRESETS.get(style, STYLE_PRESETS["none"])
    
    enhanced_prompt = prompt + preset["suffix"]
    
    # Combine negative prompts
    negatives = [DEFAULT_NEGATIVE_PROMPT]
    if preset["negative"]:
        negatives.append(preset["negative"])
    if negative_prompt:
        negatives.append(negative_prompt)
    
    enhanced_negative = ", ".join(negatives)
    
    return enhanced_prompt, enhanced_negative

def get_timestamp() -> str:
    """Return current timestamp string"""
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def get_date_folder() -> str:
    """Return date-based folder name"""
    return datetime.now().strftime("%Y-%m-%d")

def sanitize_filename(text: str, max_length: int = 50) -> str:
    """
    Convert text to safe filename
    
    Args:
        text: Input text
        max_length: Maximum filename length
        
    Returns:
        Sanitized filename string
    """
    # Remove special characters
    text = re.sub(r'[^\w\s-]', '', text)
    # Replace spaces with underscores
    text = re.sub(r'\s+', '_', text)
    # Truncate to max length
    text = text[:max_length]
    # Convert to lowercase
    text = text.lower()
    
    return text

def validate_seed(seed_input: str) -> int:
    """
    Validate and convert seed input
    
    Args:
        seed_input: Seed as string
        
    Returns:
        Valid integer seed
    """
    try:
        seed = int(seed_input)
        if seed < 0:
            seed = abs(seed)
        return seed
    except (ValueError, TypeError):
        return int(datetime.now().timestamp())

def get_prompt_engineering_tips() -> list:
    """Return list of prompt engineering tips"""
    return [
        "Be specific and descriptive",
        "Include style, lighting, and composition details",
        "Use artist names or art movements for specific styles",
        "Add quality boosters: 'highly detailed', '4K', 'professional'",
        "Separate concepts with commas",
        "Put most important elements first",
        "Use negative prompts to avoid unwanted elements",
        "Experiment with guidance scale (7-15 for balanced results)",
        "Higher steps (30-50) = better quality but slower",
        "Keep aspect ratios reasonable (512x512, 512x768, 768x512)"
    ]
