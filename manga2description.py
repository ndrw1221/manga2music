import argparse
from pathlib import Path
import pdb


def parse_args():
    parser = argparse.ArgumentParser(description="Convert manga to description")
    parser.add_argument(
        "--manga-paths",
        type=str,
        help="Path to folder containing manga images",
        default="./samples",
    )
    parser.add_argument(
        "--output-path", type=str, help="Path to output folder", default="./output"
    )
    parser.add_argument(
        "--model",
        type=str,
        help="Which LLM to use.",
        choices=["gpt-4o", "gpt-4o-mini", "llava"],
        default="llava",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    image_paths = list(Path(args.manga_paths).glob("*.jpg"))
    model = args.model
    print(f"Using model: {model}")

    if model in ["gpt-4o", "gpt-4o-mini"]:
        from models.gpt4o import GPT4o

        pdb.set_trace()
        gpt4o = GPT4o(model=model)
        descriptions = gpt4o.generate_music_description(image_paths[3:5])
        print(descriptions)

    elif model == "llava":
        from models.llava import LLAVA

        llava = LLAVA()
        descriptions = llava.generate_music_description(image_paths[1:2])
        print(descriptions)


if __name__ == "__main__":
    main()
