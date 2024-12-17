from llava.model.builder import load_pretrained_model
from llava.mm_utils import (
    process_images,
    tokenizer_image_token,
)
from llava.constants import (
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
)
from llava.conversation import conv_templates

from PIL import Image
import copy
import torch
import warnings
import json

warnings.filterwarnings("ignore")


class LLAVA:
    def __init__(
        self, pretrained_model="lmms-lab/llava-next-interleave-qwen-0.5b", device="cuda"
    ):
        self.device = device
        self.model_name = "llava_qwen"
        self.device_map = "auto"

        # Load model configuration
        self.llava_model_args = {"multimodal": True}
        self.overwrite_config = {"image_aspect_ratio": "pad"}
        self.llava_model_args["overwrite_config"] = self.overwrite_config

        # Load the model and tokenizer
        self.tokenizer, self.model, self.image_processor, self.max_length = (
            load_pretrained_model(
                pretrained_model,
                None,
                self.model_name,
                device_map=self.device_map,
                **self.llava_model_args,
            )
        )
        self.model = self.model.to(self.device)
        self.model.eval()

        # Load conversation template
        self.conv_template = "qwen_1_5"

        self._prompt_path = "prompt.json"

    def _load_images(self, image_paths):
        """Load images from file paths and process them into tensors."""
        images = [Image.open(image_path) for image_path in image_paths]
        image_tensors = process_images(images, self.image_processor, self.model.config)
        image_tensors = [
            _image.to(dtype=torch.float16, device=self.device)
            for _image in image_tensors
        ]
        image_sizes = [image.size for image in images]
        return image_tensors, image_sizes

    def _get_prompt(self):
        with open(self._prompt_path, "r") as prompt_file:
            data = json.load(prompt_file)

        return data["llava_prompt"]

    def generate_music_description(self, image_paths):
        """Generate a description based on the given series of images."""
        # Load and process images
        image_tensors, image_sizes = self._load_images(image_paths)
        prompt = self._get_prompt()
        # Prepare interleaved text-image input
        image_tokens = f"{DEFAULT_IMAGE_TOKEN}" * len(image_paths)
        question = f"{image_tokens} {prompt}"

        # Initialize conversation
        conv = copy.deepcopy(conv_templates[self.conv_template])
        conv.append_message(conv.roles[0], question)
        conv.append_message(conv.roles[1], None)
        prompt_question = conv.get_prompt()

        # Prepare input for the model
        input_ids = (
            tokenizer_image_token(
                prompt_question, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
            )
            .unsqueeze(0)
            .to(self.device)
        )

        # Generate response
        with torch.no_grad():
            cont = self.model.generate(
                input_ids,
                images=image_tensors,
                image_sizes=image_sizes,
                do_sample=True,
                temperature=0.7,
                max_new_tokens=4096,
            )
        text_outputs = self.tokenizer.batch_decode(cont, skip_special_tokens=True)
        return text_outputs[0]
