from pathlib import Path
import torch


def generate_descriptions_from_manga(
    manga_path, output_path, model, save_gpt_artifact=False
):
    """
    Generate music descriptions from manga images.

    Args:
        manga_path (str): Path to the folder containing manga images.
        output_path (str): Path to the output folder.
        model (str): Model to use for generation ('gpt-4o', 'gpt-4o-mini', 'llava-7b', 'llava-0.5b').
        save_gpt_artifact (bool): Whether to save GPT artifacts (only for 'gpt-4o' or 'gpt-4o-mini').

    Returns:
        str: The path to the saved description file.
    """
    try:
        image_paths = list(Path(manga_path).glob("*.jpg"))
        if not image_paths:
            raise ValueError(f"No images found in {manga_path}!")

        print(f"Using model: {model}")

        if model in ["gpt-4o", "gpt-4o-mini"]:
            from models.gpt4o import GPT4o

            gpt4o = GPT4o(model=model)
            descriptions = gpt4o.generate_music_description(
                image_paths, save_gpt_artifact
            )

        elif model in ["llava-7b", "llava-0.5b"]:
            from models.llava import LLAVA

            llava_model = (
                "lmms-lab/llava-next-interleave-qwen-7b"
                if model == "llava-7b"
                else "lmms-lab/llava-next-interleave-qwen-0.5b"
            )
            llava = LLAVA(pretrained_model=llava_model)
            descriptions = llava.generate_music_description(image_paths)

        # Save the descriptions to output path
        Path(output_path).mkdir(parents=True, exist_ok=True)
        file_name = f"{Path(manga_path).name}_{model}.txt"
        output_file = Path(output_path) / file_name
        with open(output_file, "w") as f:
            f.write(descriptions)

        print(f"Descriptions saved to {output_file}")
        return str(output_file)
    finally:
        if model in ["llava-7b", "llava-0.5b"]:
            print("Releasing GPU memory...")
            del llava
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            print("GPU memory released.")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Convert manga to description")
    parser.add_argument(
        "--manga-path",
        type=str,
        default="./samples",
        help="Path to folder containing manga images",
    )
    parser.add_argument(
        "--output-path", type=str, default="./output", help="Path to output folder"
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=["gpt-4o", "gpt-4o-mini", "llava-7b", "llava-0.5b"],
        default="llava-0.5b",
    )
    parser.add_argument(
        "--save-gpt-artifact",
        action="store_true",
        help="Save GPT artifacts (only for gpt-4o or gpt-4o-mini)",
    )
    args = parser.parse_args()

    try:
        generate_descriptions_from_manga(
            args.manga_path, args.output_path, args.model, args.save_gpt_artifact
        )
    except ValueError as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
