import torch
from diffusers import FluxPipeline
import os
from datetime import datetime
from PIL import Image, ImageDraw, ImageFont
from transformers import CLIPProcessor, CLIPModel
import warnings


warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Loading pipeline
pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-schnell", torch_dtype=torch.bfloat16)
pipe.enable_model_cpu_offload()  

# CLIP model and processor for evaluation
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32", clean_up_tokenization_spaces=True)

def generate_image(prompt):
    print(f"Generating image for prompt: {prompt}")

    # Generate the image
    image = pipe(
        prompt,
        guidance_scale=7.5,
        num_inference_steps=50,
        max_sequence_length=512,
        generator=torch.Generator("cpu").manual_seed(1234)
    ).images[0]

    # Converting image to RGB mode if not already
    if image.mode != "RGB":
        image = image.convert("RGB")

    # Evaluates generated image using CLIP
    inputs = clip_processor(text=[prompt], images=image, return_tensors="pt", padding=True)
    outputs = clip_model(**inputs)
    logits_per_image = outputs.logits_per_image  
    clip_score = logits_per_image.item()  # Convert tensor to a scalar

    # Add the CLIP score to the bottom of the image
    draw = ImageDraw.Draw(image)

    
    font_size = max(20, image.width // 20)  

    # Try loading the custom font; if it fails, use the default font
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", font_size)
    except IOError:
        font = ImageFont.load_default()

   
    clip_text = f"CLIP Score: {clip_score:.4f}"
    text_width, text_height = draw.textsize(clip_text, font=font)
    image_width, image_height = image.size
    new_image_height = image_height + text_height + 20  

    # Create a new image with extra space for the text
    new_image = Image.new("RGB", (image_width, new_image_height), color=(255, 255, 255))
    new_image.paste(image, (0, 0))

    # Draw the CLIP score at the bottom
    draw = ImageDraw.Draw(new_image)
    text_position = ((image_width - text_width) // 2, image_height + 10)
    draw.text(text_position, clip_text, fill="black", font=font)

    # Create a folder and filename using the first three words of the prompt
    prompt_hint = "_".join(prompt.split()[:3]) 
    output_dir = os.path.join("flux", prompt_hint)
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{timestamp}_{prompt_hint}.png"

    # Save the image with the CLIP score at the bottom
    new_image.save(os.path.join(output_dir, filename))
    print(f"Image saved as {filename} in the {output_dir} directory.")
    print(f"CLIP score (how well the image matches the prompt): {clip_score:.4f}")

def main():
    choice = input("Choose an option:\n1. Run prompts from file\n2. Enter a prompt manually\nEnter choice (1 or 2): ").strip()

    if choice == '1':
        prompts_file_path = "prompts.txt"
        
        # Read the prompts from the file
        with open(prompts_file_path, 'r') as file:
            prompts = file.readlines()

      
        prompts = [prompt.strip() for prompt in prompts if prompt.strip()]

        if not prompts:
            print("No valid prompts found in the file. Exiting.")
            return

        for prompt in prompts:
            generate_image(prompt)

    elif choice == '2':
        prompt = input("Please enter your prompt to generate an image: ").strip()
        if not prompt:
            print("Empty prompt entered. Exiting.")
            return

        generate_image(prompt)

    else:
        print("Invalid choice. Exiting.")

if __name__ == "__main__":
    main()
