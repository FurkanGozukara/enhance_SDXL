import gradio as gr
from gradio_imageslider import ImageSlider
import torch
import random

torch.jit.script = lambda f: f
from hidiffusion import apply_hidiffusion
from diffusers import (
    ControlNetModel,
    StableDiffusionXLControlNetImg2ImgPipeline,
    DDIMScheduler,
)
from controlnet_aux import AnylineDetector
from compel import Compel, ReturnedEmbeddingsType
from PIL import Image
import os
import time
import numpy as np
import sys
import platform 

IS_SPACES_ZERO = os.environ.get("SPACES_ZERO_GPU", "0") == "1"
IS_SPACE = os.environ.get("SPACE_ID", None) is not None

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16

print(f"device: {device}")
print(f"dtype: {dtype}")

def open_folder():
    open_folder_path = os.path.abspath("outputs")
    if platform.system() == "Windows":
        os.startfile(open_folder_path)
    elif platform.system() == "Linux":
        os.system(f'xdg-open "{open_folder_path}"')


model = "stabilityai/stable-diffusion-xl-base-1.0"
scheduler = DDIMScheduler.from_pretrained(model, subfolder="scheduler")
controlnet = ControlNetModel.from_pretrained(
    "TheMistoAI/MistoLine",
    torch_dtype=torch.float16,
    revision="refs/pr/3",
    variant="fp16",
)
pipe = StableDiffusionXLControlNetImg2ImgPipeline.from_pretrained(
    model,
    controlnet=controlnet,
    torch_dtype=dtype,
    variant="fp16",
    use_safetensors=True,
    scheduler=scheduler,
)

compel = Compel(
    tokenizer=[pipe.tokenizer, pipe.tokenizer_2],
    text_encoder=[pipe.text_encoder, pipe.text_encoder_2],
    returned_embeddings_type=ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED,
    requires_pooled=[False, True],
)
pipe = pipe.to(device)

if not IS_SPACES_ZERO:
    apply_hidiffusion(pipe)
    pipe.enable_model_cpu_offload()
pipe.enable_vae_tiling()

anyline = AnylineDetector.from_pretrained(
    "TheMistoAI/MistoLine", filename="MTEED.pth", subfolder="Anyline"
).to(device)


def pad_image(image):
    w, h = image.size
    if w == h:
        return image
    elif w > h:
        new_image = Image.new(image.mode, (w, w), (0, 0, 0))
        pad_h = (w - h) // 2
        new_image.paste(image, (0, pad_h))
        return new_image
    else:
        new_image = Image.new(image.mode, (h, h), (0, 0, 0))
        pad_w = (h - w) // 2
        new_image.paste(image, (pad_w, 0))
        return new_image


def predict(
    input_image,
    prompt,
    negative_prompt,
    seed,
    random_seed,
    num_inference_steps,
    guidance_scale=8.5,
    scale=2,
    controlnet_conditioning_scale=0.5,
    strength=1.0,
    controlnet_start=0.0,
    controlnet_end=1.0,
    guassian_sigma=2.0,
    intensity_threshold=3,
    progress=gr.Progress(track_tqdm=True),
):
    if IS_SPACES_ZERO:
        apply_hidiffusion(pipe)
    if input_image is None:
        raise gr.Error("Please upload an image.")
    padded_image = pad_image(input_image).resize((1024, 1024)).convert("RGB")
    conditioning, pooled = compel([prompt, negative_prompt])
    
    if random_seed:
        seed = random.randint(0, 2**32 - 1)
    
    generator = torch.manual_seed(seed)
    last_time = time.time()
    anyline_image = anyline(
        padded_image,
        detect_resolution=1280,
        guassian_sigma=max(0.01, guassian_sigma),
        intensity_threshold=intensity_threshold,
    )

    images = pipe(
        image=padded_image,
        control_image=anyline_image,
        strength=strength,
        prompt_embeds=conditioning[0:1],
        pooled_prompt_embeds=pooled[0:1],
        negative_prompt_embeds=conditioning[1:2],
        negative_pooled_prompt_embeds=pooled[1:2],
        width=1024 * scale,
        height=1024 * scale,
        controlnet_conditioning_scale=float(controlnet_conditioning_scale),
        controlnet_start=float(controlnet_start),
        controlnet_end=float(controlnet_end),
        generator=generator,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        eta=1.0,
    )
    print(f"Time taken: {time.time() - last_time}")
    os.makedirs('outputs', exist_ok=True)
    
    existing_files = os.listdir('outputs')
    existing_numbers = [int(file.split('.')[0].split('_')[1]) for file in existing_files if file.endswith('.png')]
    latest_number = max(existing_numbers) if existing_numbers else 0
    
    image_number = latest_number + 1
    filename = f'img_{image_number:05d}.png'
    filepath = os.path.join('outputs', filename)
    images.images[0].save(filepath)
    return (padded_image, images.images[0]), padded_image, anyline_image, seed


css = """
#intro{
    # max-width: 32rem;
    # text-align: center;
    # margin: 0 auto;
}
"""

with gr.Blocks(css=css) as demo:
    gr.Markdown(
        """
# Enhance This - [HiDiffusion](https://github.com/megvii-research/HiDiffusion) SDXL - V1
# Latest version of this APP always on : https://www.patreon.com/posts/104816746
        """,
        elem_id="intro",
    )
    with gr.Row():
        with gr.Column(scale=1):
            image_input = gr.Image(type="pil", label="Input Image", height=600)
            prompt = gr.Textbox(
                label="Prompt",
                info="The prompt is very important to get the desired results. Please try to describe the image as best as you can. Accepts Compel Syntax",
            )
            negative_prompt = gr.Textbox(
                label="Negative Prompt",
                value="blurry, ugly, duplicate, poorly drawn, deformed, mosaic",
            )
            with gr.Row():
                seed = gr.Slider(
                    minimum=0,
                    maximum=2**32 - 1,
                    value=1415926535,
                    step=1,
                    label="Seed",
                )
                random_seed = gr.Checkbox(label="Random Seed", value=True)
            num_inference_steps = gr.Slider(
                minimum=1,
                maximum=100,
                value=30,
                step=1,
                label="Number of Inference Steps",
            )
            with gr.Accordion(label="Advanced", open=True):
                guidance_scale = gr.Slider(
                    minimum=0,
                    maximum=50,
                    value=8.5,
                    step=0.001,
                    label="Guidance Scale",
                )
                scale = gr.Slider(
                    minimum=1,
                    maximum=5,
                    value=2,
                    step=1,
                    label="Magnification Scale",
                    interactive=not IS_SPACE,
                )
                controlnet_conditioning_scale = gr.Slider(
                    minimum=0,
                    maximum=1,
                    step=0.001,
                    value=0.5,
                    label="ControlNet Conditioning Scale",
                )
                strength = gr.Slider(
                    minimum=0,
                    maximum=1,
                    step=0.001,
                    value=1,
                    label="Strength",
                )
                controlnet_start = gr.Slider(
                    minimum=0,
                    maximum=1,
                    step=0.001,
                    value=0.0,
                    label="ControlNet Start",
                )
                controlnet_end = gr.Slider(
                    minimum=0.0,
                    maximum=1.0,
                    step=0.001,
                    value=1.0,
                    label="ControlNet End",
                )
                guassian_sigma = gr.Slider(
                    minimum=0.01,
                    maximum=10.0,
                    step=0.1,
                    value=2.0,
                    label="(Anyline) Guassian Sigma",
                )
                intensity_threshold = gr.Slider(
                    minimum=0,
                    maximum=255,
                    step=1,
                    value=3,
                    label="(Anyline) Intensity Threshold",
                )

            btn = gr.Button()
        with gr.Column(scale=2):
            with gr.Group():
                image_slider = ImageSlider(position=0.5)
            with gr.Row():
                padded_image = gr.Image(type="pil", label="Padded Image", height=600)
                anyline_image = gr.Image(type="pil", label="Anyline Image", height=600)
            with gr.Row():
                btn_open_outputs = gr.Button("Open Outputs Folder")
                btn_open_outputs.click(fn=open_folder)
    inputs = [
        image_input,
        prompt,
        negative_prompt,
        seed,
        random_seed,
        num_inference_steps,
        guidance_scale,
        scale,
        controlnet_conditioning_scale,
        strength,
        controlnet_start,
        controlnet_end,
        guassian_sigma,
        intensity_threshold,
    ]
    outputs = [image_slider, padded_image, anyline_image, seed]
    btn.click(lambda x: None, inputs=None, outputs=image_slider).then(
        fn=predict, inputs=inputs, outputs=outputs
    )
    gr.Examples(
        fn=predict,
        inputs=inputs,
        outputs=outputs,
        examples=[
            [
                "examples/lara.png",
                "photography of lara croft 8k high definition award winning",
                "blurry, ugly, duplicate, poorly drawn, deformed, mosaic",
                5436236241,
                False,
                30,
                8.5,
                2,
                0.8,
                1.0,
                0.0,
                0.9,
                2,
                3,
            ],
            [
                "examples/cybetruck.png",
                "photo of tesla cybertruck futuristic car 8k high definition on a sand dune in mars, future",
                "blurry, ugly, duplicate, poorly drawn, deformed, mosaic",
                383472451451,
                False,
                30,
                8.5,
                2,
                0.8,
                0.8,
                0.0,
                0.9,
                2,
                3,
            ],
            [
                "examples/jesus.png",
                "a photorealistic painting of Jesus Christ, 4k high definition",
                "blurry, ugly, duplicate, poorly drawn, deformed, mosaic",
                13317204146129588000,
                False,
                30,
                8.5,
                2,
                0.8,
                0.8,
                0.0,
                0.9,
                2,
                3,
            ],
            [
                "examples/anna-sullivan-DioLM8ViiO8-unsplash.png",
                "A crowded stadium with enthusiastic fans watching a daytime sporting event, the stands filled with colorful attire and the sun casting a warm glow",
                "blurry, ugly, duplicate, poorly drawn, deformed, mosaic",
                5623124123512,
                False,
                30,
                8.5,
                2,
                0.8,
                0.8,
                0.0,
                0.9,
                2,
                3,
            ],
            [
                "examples/img_aef651cb-2919-499d-aa49-6d4e2e21a56e_1024.png",
                "a large red flower on a black background 4k high definition",
                "blurry, ugly, duplicate, poorly drawn, deformed, mosaic",
                23123412341234,
                False,
                30,
                8.5,
                2,
                0.8,
                0.8,
                0.0,
                0.9,
                2,
                3,
            ],
            [
                "examples/huggingface.png",
                "photo realistic huggingface human emoji costume, round, yellow, (human skin)+++ (human texture)+++",
                "blurry, ugly, duplicate, poorly drawn, deformed, mosaic, emoji cartoon,  drawing, pixelated",
                12312353423,
                False,
                30,
                15.206,
                2,
                0.364,
                0.8,
                0.0,
                0.9,
                2,
                3,
            ],
        ],
        cache_examples="lazy",
    )


demo.queue()
demo.launch(inbrowser=True)