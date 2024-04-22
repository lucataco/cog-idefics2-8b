# Prediction interface for Cog ⚙️
# https://cog.run/python

from cog import BasePredictor, Input, Path
import os
import time
import torch
import subprocess
from PIL import Image
from transformers import AutoProcessor, Idefics2ForConditionalGeneration

MODEL_ID = "HuggingFaceM4/idefics2-8b"
MODEL_CACHE = "checkpoints"
MODEL_URL = "https://weights.replicate.delivery/default/huggingfaceh4/idefics2-8b/model.tar"

def download_weights(url, dest):
    start = time.time()
    print("downloading url: ", url)
    print("downloading to: ", dest)
    subprocess.check_call(["pget", "-x", url, dest], close_fds=False)
    print("downloading took: ", time.time() - start)

class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        # Download weights via pget
        if not os.path.exists(MODEL_CACHE):
            download_weights(MODEL_URL, MODEL_CACHE)
        self.processor = AutoProcessor.from_pretrained(
            MODEL_CACHE,
        )
        self.model = Idefics2ForConditionalGeneration.from_pretrained(
            MODEL_CACHE,
            torch_dtype=torch.bfloat16,
        ).to('cuda')

    def predict(
        self,
        image: Path = Input(description="Grayscale input image"),
        prompt: str = Input(
            description="Imput prompt", default="What is this?"
        ),
        max_new_tokens: int = Input(
            description="Maximum number of tokens to generate", default=512, ge=8, le=1024
        ),
        repetition_penalty: float = Input(
            description="Repetition penalty", default=1.2, ge=0.01, le=5.0
        ),
    ) -> str:
        """Run a single prediction on the model"""
        messages = [
            {
                "role": "user",
                "content": [{"type": "image"}] + [
                    {"type": "text", "text": prompt}
                ]
            }
        ]
        img = Image.open(image).convert("RGB")
        prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = self.processor(text=prompt, images=[img], return_tensors="pt")
        inputs = {k: v.to("cuda") for k, v in inputs.items()}
        args = {
            "max_new_tokens": max_new_tokens,
            "repetition_penalty": repetition_penalty,
            "do_sample": False
        }
        args.update(inputs)
        
        # Generate
        generated_ids = self.model.generate(**args)
        generated_texts = self.processor.batch_decode(generated_ids[:, args["input_ids"].size(1):], skip_special_tokens=True)
        print("INPUT:", prompt, "|OUTPUT:", generated_texts)
        return generated_texts[0]
    