import streamlit as st
import torch
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from peft import PeftModel
from qwen_vl_utils import process_vision_info
from PIL import Image
import os

# Streamlit Page Config
st.set_page_config(page_title="Qwen2-VL LoRA Adapter", layout="wide")
st.title("Qwen2-VL-2B-Instruct + Custom LoRA Adapter")

# Helper function to load model
@st.cache_resource
def load_model():
    base_model_name = "Qwen/Qwen2-VL-2B-Instruct"
    adapter_path = "./" # Current directory where the adapter files are

    # Determine device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Optional: For CPU / limited memory, we load in float32 or bfloat16 if supported
    torch_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32

    model = Qwen2VLForConditionalGeneration.from_pretrained(
        base_model_name,
        torch_dtype=torch_dtype,
        device_map="auto" if device == "cuda" else None,
    )

    if device == "cpu":
        model.to(device)

    # Load Adapter
    model = PeftModel.from_pretrained(model, adapter_path)
    
    # Load Processor
    processor = AutoProcessor.from_pretrained(adapter_path)
    
    return model, processor, device

with st.spinner("Loading model... this may take a while depending on your hardware."):
    try:
        model, processor, device = load_model()
        st.success("Model loaded successfully!")
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.stop()

# UI Layout
col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("### Input")
    uploaded_file = st.file_uploader("Upload an Image...", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_container_width=True)

    prompt = st.text_input("Enter your prompt:", value="Extract the markdown from this image.")
    generate_btn = st.button("Generate Markdown", type="primary")

with col2:
    st.markdown("### Output")
    if generate_btn:
        if uploaded_file is None:
            st.warning("Please upload an image first.")
        elif not prompt:
            st.warning("Please enter a prompt.")
        else:
            with st.spinner("Generating response..."):
                try:
                    # Format messages for Qwen2-VL
                    messages = [
                        {
                            "role": "user",
                            "content": [
                                {"type": "image", "image": image},
                                {"type": "text", "text": prompt},
                            ],
                        }
                    ]
                    
                    # Preparation for inference
                    text = processor.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True
                    )
                    image_inputs, video_inputs = process_vision_info(messages)
                    
                    inputs = processor(
                        text=[text],
                        images=image_inputs,
                        videos=video_inputs,
                        padding=True,
                        return_tensors="pt",
                    ).to(device)
                    
                    # Generate
                    with torch.no_grad():
                        generated_ids = model.generate(**inputs, max_new_tokens=1024)
                        
                    # Decode output
                    generated_ids_trimmed = [
                        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                    ]
                    output_text = processor.batch_decode(
                        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
                    )
                    
                    st.markdown(output_text[0])
                except Exception as e:
                    st.error(f"Error during generation: {e}")
