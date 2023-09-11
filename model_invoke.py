from diffusers import StableDiffusionPipeline
import json

pipe = StableDiffusionPipeline.from_single_file("epicrealism_naturalSinRC1VAE.safetensors")
pipe = pipe.to("cuda")
pipe.safety_checker = None

def invoke(input_text):
    input_json = json.loads(input_text)
    prompt = input_json['prompt']
    negative_prompt = input_json['negative_prompt'] if 'negative_prompt' in input_json else ""
    steps = int(input_json['steps']) if 'steps' in input_json else 50
    image = pipe(prompt=prompt, negative_prompt=negative_prompt, num_inference_steps=steps).images[0]
    image.save("generated_image.png")
    return "generated_image.png"
