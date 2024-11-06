import argparse
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Convert manga to description")
    parser.add_argument(
        "--manga-path",
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
        choices=["gpt-4o", "gpt-4o-mini", "llava-7b", "llava-0.5b"],
        default="llava-0.5b",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    image_paths = list(Path(args.manga_path).glob("*.jpg"))
    model = args.model
    print(f"Using model: {model}")

    if model in ["gpt-4o", "gpt-4o-mini"]:
        from models.gpt4o import GPT4o

        gpt4o = GPT4o(model=model)
        descriptions = gpt4o.generate_music_description(image_paths[3:5])

    elif model in ["llava-7b", "llava-0.5b"]:
        from models.llava import LLAVA

        llava_model = (
            "lmms-lab/llava-next-interleave-qwen-7b"
            if model == "llava-7b"
            else "lmms-lab/llava-next-interleave-qwen-0.5b"
        )
        llava = LLAVA(pretrained_model=llava_model)
        descriptions = llava.generate_music_description(image_paths[1:2])

    # save the descriptions to output path
    Path(args.output_path).mkdir(parents=True, exist_ok=True)
    file_name = f"{args.manga_path}_{model}.txt"
    with open(Path(args.output_path) / file_name, "w") as f:
        f.write(descriptions)
    print(f"Descriptions saved to {args.output_path}/{file_name}")


if __name__ == "__main__":
    main()
