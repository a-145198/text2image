import streamlit as st
from PIL import Image
import io
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from src.generator import ImageGenerator
from src.storage import ImageStorage
from src.utils import apply_style_preset, STYLE_PRESETS, get_prompt_engineering_tips

# Page configuration
st.set_page_config(
    page_title="Text-to-Image Generator",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {font-size: 2.5rem; font-weight: bold; text-align: center; margin-bottom: 1rem;}
    .sub-header {font-size: 1.2rem; text-align: center; color: #666; margin-bottom: 2rem;}
    .stDownloadButton {width: 100%;}
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'generator' not in st.session_state:
    st.session_state.generator = None
if 'generated_images' not in st.session_state:
    st.session_state.generated_images = []
if 'last_metadata' not in st.session_state:
    st.session_state.last_metadata = None

# Header
st.markdown('<p class="main-header">🎨 Text-to-Image Generator</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Transform your words into stunning AI-generated images</p>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    
    # Model Configuration
    st.subheader("Model")
    model_id = st.text_input(
        "Model ID",
        value="runwayml/stable-diffusion-v1-5",
        help="HuggingFace model identifier"
    )
    
    # Initialize model button
    if st.button("🚀 Load Model", type="primary", use_container_width=True):
        with st.spinner("Loading model... This may take a minute."):
            try:
                st.session_state.generator = ImageGenerator(model_id=model_id)
                st.session_state.generator.load_model()
                device_info = st.session_state.generator.get_device_info()
                st.success(f"✅ Model loaded on {device_info['device']}")
                st.info(f"Backend: {device_info['backend']}")
            except Exception as e:
                st.error(f"❌ Error loading model: {str(e)}")
    
    st.divider()
    
    # Generation Parameters
    st.subheader("Generation Parameters")
    
    style = st.selectbox(
        "Style Preset",
        options=list(STYLE_PRESETS.keys()),
        index=0,
        help="Apply predefined style enhancements"
    )
    
    num_images = st.slider(
        "Number of Images",
        min_value=1,
        max_value=4,
        value=1,
        help="Generate multiple variations"
    )
    
    num_steps = st.slider(
        "Inference Steps",
        min_value=10,
        max_value=50,
        value=30,
        step=5,
        help="More steps = better quality but slower"
    )
    
    guidance_scale = st.slider(
        "Guidance Scale",
        min_value=1.0,
        max_value=20.0,
        value=7.5,
        step=0.5,
        help="How closely to follow the prompt"
    )
    
    col1, col2 = st.columns(2)
    with col1:
        width = st.selectbox("Width", [512, 768, 1024], index=0)
    with col2:
        height = st.selectbox("Height", [512, 768, 1024], index=0)
    
    use_seed = st.checkbox("Use specific seed")
    seed = None
    if use_seed:
        seed = st.number_input("Seed", min_value=0, value=42, step=1)
    
    st.divider()
    
    # Export Settings
    st.subheader("Export")
    image_format = st.radio("Format", ["PNG", "JPEG"], horizontal=True)
    save_to_disk = st.checkbox("Save to disk", value=True)
    
    st.divider()
    
    # Device Status
    if st.session_state.generator:
        device_info = st.session_state.generator.get_device_info()
        st.subheader("Device Status")
        st.success(f"✅ {device_info['device'].upper()}")
        st.caption(device_info['backend'])

# Main content
tab1, tab2, tab3 = st.tabs(["🎨 Generate", "💡 Tips", "ℹ️ About"])

with tab1:
    # Prompt input
    st.subheader("Enter your prompt")
    
    prompt = st.text_area(
        "Describe the image you want to generate",
        placeholder="Example: a serene Japanese garden with cherry blossoms, koi pond, stone lanterns, misty morning light, highly detailed",
        height=100,
        label_visibility="collapsed"
    )
    
    negative_prompt = st.text_area(
        "Negative Prompt (optional)",
        placeholder="Things to avoid: blurry, low quality, distorted...",
        height=60
    )
    
    # Generate button
    generate_col1, generate_col2 = st.columns([3, 1])
    with generate_col1:
        generate_button = st.button("✨ Generate Images", type="primary", use_container_width=True)
    with generate_col2:
        clear_button = st.button("🗑️ Clear", use_container_width=True)
    
    if clear_button:
        st.session_state.generated_images = []
        st.session_state.last_metadata = None
        st.rerun()
    
    # Generation logic
    if generate_button:
        if not st.session_state.generator:
            st.error("⚠️ Please load the model first (see sidebar)")
        elif not prompt.strip():
            st.warning("⚠️ Please enter a prompt")
        else:
            # Apply style preset
            enhanced_prompt, enhanced_negative = apply_style_preset(
                prompt, style, negative_prompt
            )
            
            # Show enhanced prompts
            with st.expander("📝 View Enhanced Prompts"):
                st.text_area("Enhanced Prompt", enhanced_prompt, height=100)
                st.text_area("Enhanced Negative Prompt", enhanced_negative, height=80)
            
            # Generate images
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            try:
                status_text.text("🎨 Generating images...")
                progress_bar.progress(25)
                
                images, used_seed = st.session_state.generator.generate_images(
                    prompt=enhanced_prompt,
                    negative_prompt=enhanced_negative,
                    num_images=num_images,
                    num_inference_steps=num_steps,
                    guidance_scale=guidance_scale,
                    width=width,
                    height=height,
                    seed=seed
                )
                
                progress_bar.progress(75)
                
                # Save images if requested
                saved_paths = []
                if save_to_disk:
                    status_text.text("💾 Saving images...")
                    storage = ImageStorage()
                    device_info = st.session_state.generator.get_device_info()
                    
                    metadata = storage.create_metadata(
                        prompt=enhanced_prompt,
                        negative_prompt=enhanced_negative,
                        num_inference_steps=num_steps,
                        guidance_scale=guidance_scale,
                        seed=used_seed,
                        width=width,
                        height=height,
                        model_id=model_id,
                        style=style,
                        device=device_info['device']
                    )
                    
                    saved_paths = storage.save_images(
                        images, metadata, prompt, format=image_format
                    )
                    st.session_state.last_metadata = metadata
                
                progress_bar.progress(100)
                status_text.text("✅ Generation complete!")
                
                st.session_state.generated_images = images
                
                st.success(f"✨ Successfully generated {len(images)} image(s) with seed: {used_seed}")
                if saved_paths:
                    st.info(f"📁 Saved to: {Path(saved_paths[0]).parent}")
                
            except Exception as e:
                st.error(f"❌ Error during generation: {str(e)}")
                st.exception(e)
            finally:
                progress_bar.empty()
                status_text.empty()
    
    # Display generated images
    if st.session_state.generated_images:
        st.divider()
        st.subheader("Generated Images")
        
        # Display in grid
        cols = st.columns(min(len(st.session_state.generated_images), 2))
        
        for idx, image in enumerate(st.session_state.generated_images):
            with cols[idx % 2]:
                st.image(image, use_container_width=True)
                
                # Download button
                buf = io.BytesIO()
                if image_format == "JPEG":
                    image.convert("RGB").save(buf, format="JPEG", quality=95)
                else:
                    image.save(buf, format="PNG")
                
                st.download_button(
                    label=f"⬇️ Download Image {idx+1}",
                    data=buf.getvalue(),
                    file_name=f"generated_{idx+1}.{image_format.lower()}",
                    mime=f"image/{image_format.lower()}",
                    use_container_width=True
                )

with tab2:
    st.subheader("💡 Prompt Engineering Tips")
    tips = get_prompt_engineering_tips()
    for tip in tips:
        st.markdown(f"- {tip}")
    
    st.divider()
    st.subheader("📝 Example Prompts")
    
    examples = {
        "Sci-Fi": "a futuristic cyberpunk city at night, neon lights, flying cars, rain-soaked streets, cinematic wide shot",
        "Fantasy": "an ancient magical library, floating books, glowing runes, mystical atmosphere, fantasy art",
        "Portrait": "professional portrait of a wise elderly wizard, long white beard, kind eyes, detailed facial features",
        "Landscape": "serene mountain lake at sunrise, mist over water, dramatic clouds, nature photography",
        "Abstract": "abstract geometric patterns, vibrant colors, fractal design, digital art",
        "Anime": "anime girl with purple hair, school uniform, cherry blossoms, manga style, detailed",
        "Macro": "macro photography of a dewdrop on a leaf, morning light, shallow depth of field",
        "Cinematic": "epic battle scene, dramatic lighting, dust and smoke, movie still, 8K"
    }
    
    for category, example in examples.items():
        with st.expander(f"**{category}**"):
            st.code(example, language=None)

with tab3:
    st.subheader("ℹ️ About This Project")
    
    st.markdown("""
    **Local Open-Source Text-to-Image Generator**
    
    This application uses Stable Diffusion to generate images from text prompts, running entirely locally on your machine.
    
    **Features:**
    - ✅ Runs locally on Apple Silicon (MPS), CUDA GPUs, or CPU
    - ✅ Open-source models (no API costs)
    - ✅ Style presets for various artistic styles
    - ✅ Adjustable generation parameters
    - ✅ Metadata saving for reproducibility
    - ✅ Download generated images
    
    **Technology Stack:**
    - PyTorch with MPS backend
    - Hugging Face Diffusers
    - Streamlit for UI
    - Stable Diffusion v1.5
    
    **Ethical AI Use:**
    - This tool should be used responsibly
    - Do not generate inappropriate, harmful, or copyrighted content
    - Generated images should be marked as AI-generated
    - Respect intellectual property rights
    
    **Hardware Requirements:**
    - Recommended: Apple M-series Mac or NVIDIA GPU
    - Minimum: 8GB RAM (16GB+ recommended)
    - Storage: ~5GB for model files
    """)
    
    st.divider()
    
    st.caption("Created with ❤️ using Stable Diffusion and Streamlit")
    st.caption("Model: runwayml/stable-diffusion-v1-5")
