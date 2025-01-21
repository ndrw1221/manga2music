import gradio as gr
import numpy as np
import pytz
from datetime import datetime
from manga2description import generate_descriptions_from_manga

# from description2music import generate_music_from_descriptions
from description2music import generate_music_from_text


def image_to_music_desc(images_folder, model_choice):
    timestamp = datetime.now(pytz.timezone("Asia/Taipei")).strftime("%Y%m%d_%H%M%S")
    output_path = f"./output/descriptions/{timestamp}"
    try:
        # Call the manga description generator
        description_file = generate_descriptions_from_manga(
            manga_path=images_folder,
            output_path=output_path,
            model=model_choice,
        )
        with open(description_file, "r") as f:
            descriptions = f.read()
        return descriptions, gr.update(interactive=True)
    except Exception as e:
        return f"Error: {e}", gr.update(interactive=True)


def music_desc_to_music(music_desc, model_choice, duration, audio_format, bulk_count):
    timestamp = datetime.now(pytz.timezone("Asia/Taipei")).strftime("%Y%m%d_%H%M%S")
    output_folder = f"./output/musics/{timestamp}"
    try:
        # Call the music generator
        generated_files_paths = generate_music_from_text(
            description=music_desc,
            output_folder=output_folder,
            model_name=model_choice,
            duration=duration,
            audio_format=audio_format,
            bulk_count=bulk_count,
            device="cuda",
        )

        return (
            generated_files_paths,
            gr.update(value="", visible=False),
            gr.update(interactive=True),
        )

    except Exception as e:
        return (
            f"Error: {e}",
            gr.update(value="", visible=False),
            gr.update(interactive=True),
        )


def single_stage(
    images_folder,
    img_to_desc_model,
    desc_to_music_model,
    duration,
    audio_format,
    bulk_count,
):
    try:
        # Combines Stage 1 and Stage 2
        music_desc, _ = image_to_music_desc(images_folder, img_to_desc_model)
        return music_desc_to_music(
            music_desc, desc_to_music_model, duration, audio_format, bulk_count
        )
    except Exception as e:
        return (
            f"Error: {e}",
            gr.update(value="", visible=False),
            gr.update(interactive=True),
        )


# Stage 1 GUI
with gr.Blocks() as image_to_music_gui:
    with gr.Row():
        gr.Markdown("### Stage 1 - Manga Image to Music Description")
    with gr.Row():
        images_folder_input = gr.Textbox(
            label="Select Folder Containing Manga Images",
            placeholder="Drag and drop the folder here or paste the path",
        )
        img_to_desc_model_choice = gr.Dropdown(
            value="gpt-4o-mini",
            choices=["gpt-4o", "gpt-4o-mini", "llava-7b", "llava-0.5b"],
            label="Choose Model for Image to Music Description",
        )
    with gr.Row():
        music_desc_output = gr.Textbox(
            label="Generated Music Description",
            interactive=False,
            show_copy_button=True,
        )
    with gr.Row():
        gen_desc_button = gr.Button("Generate Description", interactive=False)

    # Dynamically enable/disable the button
    images_folder_input.change(
        lambda folder: gr.update(interactive=bool(folder)),
        inputs=[images_folder_input],
        outputs=[gen_desc_button],
    )

    gen_desc_button.click(
        lambda: gr.update(interactive=False),
        inputs=[],
        outputs=[gen_desc_button],
    )

    gen_desc_button.click(
        image_to_music_desc,
        inputs=[images_folder_input, img_to_desc_model_choice],
        outputs=[music_desc_output, gen_desc_button],
    )

# Stage 2 GUI
with gr.Blocks() as music_desc_to_music_gui:
    with gr.Row():
        gr.Markdown("### Stage 2 - Music Description to Music")
    with gr.Row():
        music_desc_input = gr.Textbox(
            label="Music Description",
            placeholder="Input music description here",
        )
    with gr.Row():
        desc_to_music_model_choice = gr.Dropdown(
            value="musicgen-medium",
            choices=["musicgen-small", "musicgen-medium", "musicgen-large"],
            label="Choose Model for Music Generation",
        )
        audio_format_choice = gr.Dropdown(
            value="mp3", choices=["wav", "mp3", "ogg", "flac"], label="Audio Format"
        )
    with gr.Row():
        duration_input = gr.Slider(
            value=30, minimum=1, maximum=120, step=1, label="Audio Duration (seconds)"
        )
        bulk_count_input = gr.Slider(
            value=3, minimum=1, maximum=10, step=1, label="Bulk Generation"
        )
    with gr.Row():
        progress_bar = gr.Textbox(value="", visible=False, label="generating...")
    with gr.Row():
        gen_music_button = gr.Button("Generate Music", interactive=False)

    audio_paths = gr.State([])

    # Dynamically enable/disable the button
    music_desc_input.change(
        lambda description: gr.update(interactive=bool(description)),
        inputs=[music_desc_input],
        outputs=[gen_music_button],
    )

    # Show progress bar during generation
    gen_music_button.click(
        lambda: (
            gr.update(value="Generating music...", visible=True),
            gr.update(interactive=False),
        ),
        inputs=[],
        outputs=[progress_bar, gen_music_button],
        show_progress=True,
    )

    # Generate music
    gen_music_button.click(
        music_desc_to_music,
        inputs=[
            music_desc_input,
            desc_to_music_model_choice,
            duration_input,
            audio_format_choice,
            bulk_count_input,
        ],
        outputs=[audio_paths, progress_bar, gen_music_button],
    )

    @gr.render(inputs=audio_paths)
    def render_audio_display(audio_paths):
        if len(audio_paths) == 0:
            return
        for audio_path in audio_paths:
            gr.Audio(
                value=audio_path,
                label=audio_path.split("/")[-1],
                show_download_button=True,
            )


# Single-Stage GUI
with gr.Blocks() as single_stage_gui:
    with gr.Row():
        images_folder_input_single = gr.Textbox(
            label="Select Folder Containing Manga Images",
            placeholder="Drag and drop the folder here or paste the path",
        )
    with gr.Row():
        img_to_desc_model_choice_single = gr.Dropdown(
            value="gpt-4o-mini",
            choices=["gpt-4o", "gpt-4o-mini", "llava-7b", "llava-0.5b"],
            label="Choose Model for Image to Music Description",
        )
        desc_to_music_model_choice_single = gr.Dropdown(
            value="musicgen-medium",
            choices=["musicgen-small", "musicgen-medium", "musicgen-large"],
            label="Choose Model for Music Generation",
        )
    with gr.Row():
        duration_input_single = gr.Slider(
            value=30, minimum=1, maximum=120, step=1, label="Audio Duration (seconds)"
        )
        audio_format_choice_single = gr.Dropdown(
            value="mp3", choices=["wav", "mp3", "ogg", "flac"], label="Audio Format"
        )
        bulk_count_input_single = gr.Slider(
            value=3, minimum=1, maximum=10, step=1, label="Bulk Generation"
        )
    with gr.Row():
        progress_bar_single = gr.Textbox(value="", visible=False, label="generating...")
    with gr.Row():
        gen_single_stage_button = gr.Button("Generate Music", interactive=False)

    audio_paths_single = gr.State([])

    # Dynamically enable/disable the button
    images_folder_input_single.change(
        lambda folder: gr.update(interactive=bool(folder)),
        inputs=[images_folder_input_single],
        outputs=[gen_single_stage_button],
    )

    # Show progress bar during generation
    gen_single_stage_button.click(
        lambda: (
            gr.update(value="Generating music...", visible=True),
            gr.update(interactive=False),
        ),
        inputs=[],
        outputs=[progress_bar_single, gen_single_stage_button],
        show_progress=True,
    )

    gen_single_stage_button.click(
        single_stage,
        inputs=[
            images_folder_input_single,
            img_to_desc_model_choice_single,
            desc_to_music_model_choice_single,
            duration_input_single,
            audio_format_choice_single,
            bulk_count_input_single,
        ],
        outputs=[audio_paths_single, progress_bar_single, gen_single_stage_button],
    )

    @gr.render(inputs=audio_paths_single)
    def render_audio_display_single(audio_paths_single):
        if len(audio_paths_single) == 0:
            return
        for audio_path in audio_paths_single:
            gr.Audio(value=audio_path, label=audio_path.split("/")[-1])


# Main App with Navigation
with gr.Blocks() as app:
    with gr.Row():
        gr.Markdown("# Manga Image to Music Generator")
    with gr.Tab("Two-Stage Generation"):
        image_to_music_gui.render()
        music_desc_to_music_gui.render()
    with gr.Tab("Single-Stage Generation"):
        gr.Markdown("### Single-Stage - Image to Music")
        single_stage_gui.render()

if __name__ == "__main__":
    app.launch()
