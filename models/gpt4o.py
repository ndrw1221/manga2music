import json
import base64
import os
import pytz
from datetime import datetime
from pathlib import Path
from openai import OpenAI


class GPT4o(OpenAI):
    def __init__(self, model="gpt-4o"):
        super().__init__()
        self.model = model
        self._prompt_path = "prompt.json"
        self.timezone = pytz.timezone("Asia/Taipei")

    def _get_prompt(self):
        with open(self._prompt_path, "r") as prompt_file:
            data = json.load(prompt_file)

        first_prompt = data["gpt_first_prompt"]
        second_prompt = data["gpt_second_prompt"]

        return first_prompt, second_prompt

    def _encode_image(self, image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")

    def _analyze_images(self, image_paths):
        base64_images = [self._encode_image(_image_path) for _image_path in image_paths]
        first_prompt, _ = self._get_prompt()

        content = []
        for _base64_image in base64_images:
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{_base64_image}"},
                }
            )
        content.append(
            {
                "type": "text",
                "text": first_prompt,
            }
        )

        response = self.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": content,
                }
            ],
        )

        return response.choices[0].message.content

    def _save_artifact(self, analysis_content):
        artifacts_dir = Path("artifacts")
        artifacts_dir.mkdir(exist_ok=True)

        # Generate a timestamp-based filename
        timestamp = datetime.now(self.timezone).strftime("%Y%m%d_%H%M%S")
        artifact_file = artifacts_dir / f"artifact_{timestamp}.txt"

        with open(artifact_file, "w") as file:
            file.write(analysis_content)

        print(f"Artifact saved to {artifact_file}")

    def generate_music_description(self, image_paths, save_artifact=False):
        image_analysis = self._analyze_images(image_paths)

        if save_artifact:
            self._save_artifact(image_analysis)

        _, second_prompt = self._get_prompt()

        content = f"{second_prompt} {image_analysis}"

        response = self.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": content,
                },
            ],
        )

        return response.choices[0].message.content
