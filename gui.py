import gradio as gr
import numpy as np
import pytz
from datetime import datetime
from manga2description import generate_descriptions_from_manga

# from description2music import generate_music_from_descriptions
from description2music import generate_music_from_text


def image_to_music_desc(images_folder, model_choice):
    output_path = "./output"
    try:
        # Call the manga description generator
        description_file = generate_descriptions_from_manga(
            manga_path=images_folder,
            output_path=output_path,
            model=model_choice,
            save_gpt_artifact=(model_choice in ["gpt-4o", "gpt-4o-mini"]),
        )
        with open(description_file, "r") as f:
            descriptions = f.read()
        return descriptions
    except ValueError as e:
        return f"Error: {e}"


def music_desc_to_music(music_desc, model_choice, duration, audio_format, bulk_count):
    timestamp = datetime.now(pytz.timezone("Asia/Taipei")).strftime("%Y%m%d_%H%M%S")
    output_folder = f"./output/{timestamp}"
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

    except ValueError as e:
        return f"Error: {e}"


def single_stage(
    images_folder,
    img_to_desc_model,
    desc_to_music_model,
    duration,
    audio_format,
    bulk_count,
):
    # Combines Stage 1 and Stage 2
    music_desc = image_to_music_desc(images_folder, img_to_desc_model)
    return music_desc_to_music(
        music_desc, desc_to_music_model, duration, audio_format, bulk_count
    )


# Stage 1 GUI
with gr.Blocks() as image_to_music_gui:
    with gr.Row():
        gr.Markdown("### Manga Image to Music Description (Stage 1)")
    with gr.Row():
        images_folder_input = gr.Textbox(
            label="Select Folder Containing Manga Images",
            placeholder="Drag and drop the folder here or paste the path",
        )
        img_to_desc_model_choice = gr.Dropdown(
            choices=["gpt-4o", "gpt-4o-mini", "llava-7b", "llava-0.5b"],
            label="Choose Model for Image to Music Description",
        )
    with gr.Row():
        music_desc_output = gr.Textbox(
            label="Generated Music Description", interactive=False
        )
    with gr.Row():
        gen_desc_button = gr.Button("Generate Description")
    gen_desc_button.click(
        image_to_music_desc,
        inputs=[images_folder_input, img_to_desc_model_choice],
        outputs=music_desc_output,
    )

# Stage 2 GUI
with gr.Blocks() as music_desc_to_music_gui:
    with gr.Row():
        gr.Markdown("### Music Description to Music (Stage 2)")
    with gr.Row():
        music_desc_input = gr.Textbox(
            label="Music Description",
            placeholder="Input music description here",
        )
        desc_to_music_model_choice = gr.Dropdown(
            value="musicgen-medium",
            choices=["musicgen-small", "musicgen-medium", "musicgen-large"],
            label="Choose Model for Music Generation",
        )
    with gr.Row():
        duration_input = gr.Slider(
            value=30, minimum=1, maximum=120, step=1, label="Audio Duration (seconds)"
        )
        audio_format_choice = gr.Dropdown(
            value="mp3", choices=["wav", "mp3", "ogg", "flac"], label="Audio Format"
        )
        bulk_count_input = gr.Number(value=3, precision=0, label="Bulk Generation")
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
        show_progress=False,
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
            gr.Audio(value=audio_path, label=audio_path.split("/")[-1])


# Single-Stage GUI
with gr.Blocks() as single_stage_gui:
    with gr.Row():
        gr.Markdown("### Single-Stage Generation")
    with gr.Row():
        images_folder_input_single = gr.Textbox(
            label="Input Folder of Images", placeholder="Path to folder"
        )
        img_to_desc_model_choice_single = gr.Dropdown(
            choices=["gpt-4o", "gpt-4o-mini", "llava-7b", "llava-0.5b"],
            label="Choose Model for Image to Music Description",
        )
        desc_to_music_model_choice_single = gr.Dropdown(
            choices=["musicgen-small", "musicgen-medium", "musicgen-large"],
            label="Choose Model for Music Generation",
        )
    with gr.Row():
        duration_input_single = gr.Slider(
            minimum=1, maximum=120, step=1, label="Audio Duration (seconds)"
        )
        audio_format_choice_single = gr.Dropdown(
            choices=["wav", "mp3", "ogg", "flac"], label="Audio Format"
        )
        bulk_count_input_single = gr.Number(
            value=1, precision=0, label="Number of Variations for Bulk Generation"
        )
    with gr.Row():
        music_output_single = gr.Textbox(
            label="Generated Music Files", interactive=False
        )
    with gr.Row():
        gen_single_stage_button = gr.Button("Generate Music")
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
        outputs=music_output_single,
    )

# Main App with Navigation
with gr.Blocks() as app:
    with gr.Row():
        gr.Markdown("# Manga Image to Music Generator")
    with gr.Tab("Two-Stage Generation"):
        gr.Markdown("### Perform Stage 1 and Stage 2 Separately")
        image_to_music_gui.render()
        music_desc_to_music_gui.render()
    with gr.Tab("Single-Stage Generation"):
        gr.Markdown("### Perform Single-Stage Generation")
        single_stage_gui.render()

app.launch()
