import torch
from diffusers import StableDiffusionXLPipeline

pipe = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    custom_pipeline="multimodalart/sdxl_perturbed_attention_guidance",
    torch_dtype=torch.float16
)

device="cuda"
pipe = pipe.to(device)
print("Enter your prompt")
prompt=str(input())
output = pipe(
        "",
        num_inference_steps=50,
        guidance_scale=0.0,
        pag_scale=5.0,
        pag_applied_layers=['mid']
    ).images



output = pipe(prompt
        ,
        num_inference_steps=25,
        guidance_scale=4.0,
        pag_scale=3.0,
        pag_applied_layers=['mid']
    ).images[0]
output.save("output.png")