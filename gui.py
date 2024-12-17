import gradio as gr
from manga2description import generate_descriptions_from_manga


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
    # Placeholder function for Stage 2
    return [
        f"Generated {audio_format} audio with {model_choice}, {duration}s duration (Bulk {i+1})"
        for i in range(bulk_count)
    ]


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
        music_desc_input = gr.Textbox(label="Input Music Description")
        desc_to_music_model_choice = gr.Dropdown(
            choices=["musicgen-small", "musicgen-medium", "musicgen-large"],
            label="Choose Model for Music Generation",
        )
    with gr.Row():
        duration_input = gr.Slider(
            minimum=1, maximum=120, step=1, label="Audio Duration (seconds)"
        )
        audio_format_choice = gr.Dropdown(
            choices=["wav", "mp3", "ogg", "flac"], label="Audio Format"
        )
        bulk_count_input = gr.Number(
            value=1, precision=0, label="Number of Variations for Bulk Generation"
        )
    with gr.Row():
        music_output = gr.Textbox(label="Generated Music Files", interactive=False)
    with gr.Row():
        gen_music_button = gr.Button("Generate Music")
    gen_music_button.click(
        music_desc_to_music,
        inputs=[
            music_desc_input,
            desc_to_music_model_choice,
            duration_input,
            audio_format_choice,
            bulk_count_input,
        ],
        outputs=music_output,
    )

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
