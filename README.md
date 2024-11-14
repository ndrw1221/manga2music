# Manga2Music: Automated Music Generation from Manga Images

## Project Overview
This project aims to generate music from manga images using large language models (LLMs) and a music generation model. It follows a two-step process:

1. **Manga to Description**: Extracts descriptive text from a series of manga images using models like **GPT-4o** or **LLAVA**.
2. **Description to Music**: Converts the generated descriptions into music using the **MusicGen** model.

The resulting audio captures the mood and atmosphere of the manga content, creating a unique multimedia experience.

## Table of Contents
- [Setup and Installation](#setup-and-installation)
- [How to Use](#how-to-use)
  - [Step 1: Manga to Description](#step-1-manga-to-description)
  - [Step 2: Description to Music](#step-2-description-to-music)
- [Model Selection](#model-selection)


## Setup and Installation
### Prerequisites
- Python 3.9
- CUDA-compatible GPU (for faster processing)
- `conda` (recommneded) or `pip` for package installation
- OpenAI API Key (required for GPT-4o and GPT-4o-mini models)

### Installation
1. Clone this repository:

   ```bash
   git clone https://github.com/ndrw1221/manga2music.git
   cd manga-to-music
   ```

2. Install the required Python packages:

    Using **Ananconda** or **Miniconda** (recommneded):
    ```bash
    conda create --file environment.yml
    ```

    Using **pip**:
    ```bash
    pip install -r requirements.txt
    ```

3. Set up your OpenAI API Key:
    
    You need to have an OpenAI API key to use the GPT-4o model. Export it as an environment variable:
    ```bash
    export OPENAI_API_KEY="your_openai_api_key_here"
    ```
    
    You can add this line to your shell profile (e.g., ~/.bashrc, ~/.zshrc) to avoid setting it every time:
    ```bash
    echo 'export OPENAI_API_KEY="your_openai_api_key_here"' >> ~/.bashrc
    source ~/.bashrc
    ```
4. Validate installation:

    To validate you have a successfull installation, run:

    ```bash
    python manga2description.py
    ```
    You should see `samples_llava-0.5b.txt` under `./output`.
    
    Then, run:
    ```bash
    python description2music.py
    ```
    You should see `samples_llava-0.5b.wav` under `.output`.

## How to Use

The process involves two main scripts:

### Step 1: Manga to Description

This step extracts descriptions from manga images using the manga2description.py script.

**Usage**

```bash
python manga2description.py \
--output-path ./output \
--model llava-0.5b
```

**Arguments**

- `--manga-path`: Path to the folder containing manga images (default: ./samples).
- `--output-path`: Path to save the generated descriptions (default: ./output).
- `--model`: Model to use for description generation (default: llava-0.5b; options: gpt-4o, gpt-4o-mini, llava-7b, llava-0.5b).
- `--save-gpt-artifact`: Save GPT artifacts (only applicable for gpt-4o or gpt-4o-mini models).

**Note**

For `gpt-4o` and `gpt-4o-mini` models, we leverage a two-stage prompting strategy: first, the model analyzes the input image(s) across four aspects—**Genre**, **Emotional Atmosphere**, **Focus**, and **Plot**—to extract key details; then, based on the analysis, the model generates the music description. The `--save-gpt-artifact` flag saves the results of the first stage analysis to an artifact file, which is useful for reviewing intermediate outputs before generating the final music description. This flag is only applicable when using `gpt-4o` or `gpt-4o-mini`.

### Step 2: Description to Music

After generating descriptions, convert them into music using the description2music.py script.

**Usage**

```bash
python description2music.py \
--description-path ./output \
--output-path ./output \
--model musicgen-large \
--duration 10 \
--audio-format wav \
--device cuda
```

**Arguments**

- `--description-path`: Path to the folder containing text descriptions (default: ./output).
- `--output-path`: Path to save the generated music (default: ./output).
- `--model`: MusicGen model to use (default: musicgen-large; options: musicgen-small, musicgen-medium, musicgen-large).
- `--duration`: Length of the generated music in seconds (default: 10).
- `--audio-format`: Audio format to save the generated music (default: wav; options: wav, mp3, ogg, flac).
- `--device`: Device to run the model on (cuda or cpu).

 ### Notes
- The model can take images of arbitrary sizes, so it is not necessary to cut input images into fixed sizes before processing. This allows for greater flexibility when using different manga sources.

- A single description will be generated for all images within a single `--manga-path`, and one single piece of music will be generated from that description. If you wish to generate multiple pieces of music for different sections of the manga, organize the images by placing all the images belonging to the same section into separate folders.

## Model Selection

This project uses two categories of models: **text generation models** for manga description extraction and **music generation models** for converting text descriptions into music. Below, we provide an overview of the models used, along with their trade-offs, to help you make an informed decision based on your use case.

### 1. Manga to Description Models
This step relies on Large Language Models (LLMs) to extract meaningful descriptions from manga images. The models used include **GPT-4o**, **GPT-4o-mini**, **LLAVA-7b**, and **LLAVA-0.5b**.

| Model          | Size    | Strengths                                                                | Limitations                                                            |
|----------------|---------|--------------------------------------------------------------------------|------------------------------------------------------------------------|
| **GPT-4o**     | Large   | Superior text understanding and generation; highly detailed descriptions | Requires an OpenAI API key; higher cost per request                    |
| **GPT-4o-mini**| Medium  | Good balance between performance and cost                                | Requires an OpenAI API key; slightly less accurate than GPT-4o, especially in nuanced contexts     |
| **LLAVA-7b**   | Large   | Strong at multimodal understanding; good for complex scenes              | Requires more computational resources (GPU with >16GB VRAM recommended)|
| **LLAVA-0.5b** | Small   | Lightweight and efficient; faster processing                             | Limited in complex scene understanding and descriptive accuracy        |

#### **Model Selection Considerations**
- **Accuracy vs. Cost**: If you prioritize accuracy and have sufficient budget, **GPT-4o** is ideal due to its high-quality text generation. However, if you're working with limited API budgets, **GPT-4o-mini** is a more cost-effective option.
- **Computational Resources**: **LLAVA** models are open-source and can run on local machines, which is great for users who want to avoid API costs. However, they require significant GPU resources, especially **LLAVA-7b**. For users with limited hardware capabilities, **LLAVA-0.5b** is a faster, more lightweight option.
- **Use Case**: For simpler manga scenes, **LLAVA-0.5b** is sufficient. For more complex scenes where detailed storytelling is needed, opt for **LLAVA-7b** or **GPT-4o**.

---

### 2. Description to Music Models
The second step uses **MusicGen** models to generate music from the textual descriptions produced in the first step. Available models include **musicgen-small**, **musicgen-medium**, and **musicgen-large**.

| Model               | Size      | Strengths                                                      | Limitations                                                               |
|---------------------|-----------|----------------------------------------------------------------|---------------------------------------------------------------------------|
| **musicgen-small**  | Small     | Fast generation; lower computational requirements              | Limited complexity and audio quality                                      |
| **musicgen-medium** | Medium    | Good balance between quality and speed                         | Requires more GPU memory (8GB VRAM recommended)                           |
| **musicgen-large**  | Large     | Highest audio fidelity; suitable for complex compositions      | Slowest processing; requires substantial GPU memory (16GB VRAM or higher) |

#### **Model Selection Considerations**
- **Speed vs. Quality**: If you prioritize speed and are willing to compromise on audio quality, **musicgen-small** is your best choice. This model is efficient on consumer-grade GPUs and even CPUs.
- **Quality for Professional Use**: If your project demands higher audio fidelity and more nuanced compositions (e.g., for video game soundtracks or anime), **musicgen-large** provides the most sophisticated output but requires significant computational resources.
- **Balancing Trade-offs**: For general use cases where quality is important but resources are limited, **musicgen-medium** offers a great middle ground.

---

### Summary of Recommendations
| Scenario                                                    | Recommended Models                                |
|-------------------------------------------------------------|---------------------------------------------------|
| **High quality text & music output, no budget constraints** | GPT-4o + musicgen-large                           |
| **Moderate budget, good quality output**                    | GPT-4o-mini + musicgen-medium                     |
| **Limited GPU resources, faster generation**                | LLAVA-0.5b + musicgen-small                       |
| **Open-source solution, high-quality descriptions**         | LLAVA-7b + musicgen-large                         |

By understanding the trade-offs, you can optimize the model selection based on your available resources, budget, and the specific requirements of your project.