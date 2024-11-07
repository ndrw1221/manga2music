import argparse
import scipy.io.wavfile as wavfile
from pathlib import Path
from audiocraft.models import MusicGen
from audiocraft.data.audio import audio_write


def parse_args():
    parser = argparse.ArgumentParser(description="Convert description to music")
    parser.add_argument(
        "--description-path",
        type=str,
        help="Path to folder containing descriptions",
        default="./output",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        help="Path to output folder",
        default="./output",
    )
    parser.add_argument(
        "--model",
        type=str,
        help="Which size of the musicgen model to use.",
        choices=["musicgen-small", "musicgen-medium", "musicgen-large"],
        default="musicgen-small",
    )
    parser.add_argument(
        "--duration",
        type=int,
        help="Length of the generated music in seconds",
        default=10,
    )
    parser.add_argument(
        "--device",
        type=str,
        help="Device to run the model on",
        default="cuda",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print(f"Using model: {args.model}")
    print(f"Use device: {args.device}")

    # Load the descriptions
    description_paths = list(Path(args.description_path).glob("*.txt"))
    descriptions = []
    for description_path in description_paths:
        with open(description_path, "r") as f:
            descriptions.append(f.read())

    # Load the model
    model = MusicGen.get_pretrained(f"facebook/{args.model}", device=args.device)
    model.set_generation_params(duration=args.duration)

    # Generate music for each description
    musics = model.generate(descriptions, progress=True)

    # Save the generated music
    for music, description_path in zip(musics, description_paths):
        output_path = Path(args.output_path) / f"{description_path.stem}.wav"
        audio_write(
            output_path,
            music.cpu(),
            model.sample_rate,
            strategy="loudness",
            loudness_compressor=True,
            add_suffix=False,
        )
        print(f"Generated music for {description_path.stem} at {output_path}")


if __name__ == "__main__":
    main()
