from pathlib import Path
from audiocraft.models import MusicGen
from audiocraft.data.audio import audio_write
import numpy as np
import json
import torch


def load_model(model_name, device="cuda"):
    """
    Load the MusicGen model.

    Args:
        model_name (str): Model name ('musicgen-small', 'musicgen-medium', 'musicgen-large').
        device (str): Device to run the model on ('cuda' or 'cpu').

    Returns:
        model: The loaded MusicGen model.
    """
    print(f"Loading model: {model_name} on {device}")
    model = MusicGen.get_pretrained(f"facebook/{model_name}", device=device)
    return model


def save_audio(audio, sr, output_path, audio_format):
    """
    Save audio to a file.

    Args:
        audio (np.ndarray): Audio data.
        sr (int): Sample rate.
        output_path (str): Path to save the audio file.
        audio_format (str): Audio format ('wav', 'mp3', 'ogg', 'flac').
    """
    audio_write(
        output_path,
        audio,
        sr,
        format=audio_format,
        strategy="loudness",
        loudness_compressor=True,
        add_suffix=False,
    )


def generate_music_from_text(
    description,
    output_folder,
    model_name,
    duration,
    audio_format,
    bulk_count=1,
    device="cuda",
):
    """
    Generate music from a text description.

    Args:
        description (str): Text description for music generation.
        output_folder (str): Folder path to save the generated music.
        model_name (str): Size of the MusicGen model ('musicgen-small', 'musicgen-medium', 'musicgen-large').
        duration (int): Length of the generated music in seconds.
        audio_format (str): Audio format to save the music ('wav', 'mp3', 'ogg', 'flac').
        bulk_count (int): Number of music samples to generate.
        device (str): Device to run the model on ('cuda' or 'cpu').

    Returns:
        list of tuple: List of tuples containing the sample rate and audio tensor of each generated music.
    """
    try:
        # Load the MusicGen model
        model = load_model(model_name, device=device)
        model.set_generation_params(duration=duration)

        descriptions = [description] * bulk_count

        # Generate music from the description
        print("Generating music...")
        musics = model.generate(descriptions, progress=True)
        sr = model.sample_rate

        # Create the output folder if it doesn't exist
        Path(output_folder).mkdir(parents=True, exist_ok=True)

        # Save the generated music
        print("Saving generated music...")
        generated_files_path = []
        for i, music in enumerate(musics):
            output_path = f"{output_folder}/{i}.{audio_format}"
            save_audio(
                music.cpu(),
                sr,
                output_path,
                audio_format,
            )
            print(f"Generated music saved at: {output_path}")
            generated_files_path.append(output_path)

        # Save metadata to data.json
        metadata = {
            "description": description,
            "model_name": model_name,
            "duration": duration,
            "bulk_count": bulk_count,
            "audio_format": audio_format,
            "generated_files": generated_files_path,
        }
        metadata_path = Path(output_folder) / "data.json"
        with open(metadata_path, "w") as metadata_file:
            json.dump(metadata, metadata_file, indent=4)
        print(f"Metadata saved at: {metadata_path}")

        return generated_files_path

    finally:
        # Cleanup: Release VRAM
        print("Releasing GPU resources...")
        del model  # Delete the model
        torch.cuda.empty_cache()  # Clear PyTorch GPU cache
        torch.cuda.synchronize()  # Synchronize to ensure all operations are complete


def generate_music_from_folder_of_descriptions(
    description_path, output_path, model_name, duration, audio_format, device="cuda"
):
    """
    Generate music from descriptions using MusicGen.

    Args:
        description_path (str): Path to folder containing description files.
        output_path (str): Path to folder to save generated music.
        model_name (str): Size of the MusicGen model ('musicgen-small', 'musicgen-medium', 'musicgen-large').
        duration (int): Length of the generated music in seconds.
        audio_format (str): Audio format to save the music ('wav', 'mp3', 'ogg', 'flac').
        device (str): Device to run the model on ('cuda' or 'cpu').

    Returns:
        list: List of paths to the generated music files.
    """
    description_paths = list(Path(description_path).glob("*.txt"))
    if not description_paths:
        raise ValueError(f"No description files found in {description_path}!")

    descriptions = []
    for description_file in description_paths:
        with open(description_file, "r") as f:
            descriptions.append(f.read())

    # Load the MusicGen model
    model = load_model(model_name, device=device)
    model.set_generation_params(duration=duration)

    # Generate music from descriptions
    print("Generating music...")
    musics = model.generate(descriptions, progress=True)

    # Save generated music files
    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    generated_files = []
    for music, description_file in zip(musics, description_paths):
        output_file = output_dir / f"{description_file.stem}.{audio_format}"
        audio_write(
            output_file,
            music.cpu(),
            model.sample_rate,
            format=audio_format,
            strategy="loudness",
            loudness_compressor=True,
            add_suffix=False,
        )
        generated_files.append(str(output_file))
        print(f"Generated music saved at: {output_file}")

    return generated_files


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Convert description to music")
    parser.add_argument(
        "--description-path",
        type=str,
        default="./output",
        help="Path to folder containing descriptions",
    )
    parser.add_argument(
        "--output-path", type=str, default="./output", help="Path to output folder"
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=["musicgen-small", "musicgen-medium", "musicgen-large"],
        default="musicgen-large",
    )
    parser.add_argument(
        "--duration",
        type=int,
        default=10,
        help="Length of the generated music in seconds",
    )
    parser.add_argument(
        "--audio-format",
        type=str,
        choices=["wav", "mp3", "ogg", "flac"],
        default="wav",
        help="Audio format to save the music",
    )
    parser.add_argument(
        "--device", type=str, default="cuda", help="Device to run the model on"
    )
    parser.add_argument(
        "--debug", action="store_true", help="Enable debug mode for verbose output"
    )
    args = parser.parse_args()

    try:
        if args.debug:
            result = generate_music_from_text(
                description="A cool Jazz music.",
                output_folder="./output/debug",
                model_name="musicgen-small",
                duration=5,
                audio_format="wav",
                bulk_count=10,
                device="cuda",
            )
            print(result)
        else:
            generate_music_from_folder_of_descriptions(
                args.description_path,
                args.output_path,
                args.model,
                args.duration,
                args.audio_format,
                args.device,
            )
    except ValueError as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
