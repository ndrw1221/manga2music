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
python manga2description.py --manga-path ./samples --output-path ./output --model llava-0.5b
```

**Arguments**

- --manga-path: Path to the folder containing manga images (default: ./samples).
- --output-path: Path to save the generated descriptions (default: ./output).
- --model: Model to use for description generation (options: gpt-4o, gpt-4o-mini, llava-7b, llava-0.5b).

### Step 2: Description to Music

After generating descriptions, convert them into music using the description2music.py script.

**Usage**

```bash
python description2music.py --description-path ./output --output-path ./output --model musicgen-small --duration 10 --device cuda
```

**Arguments**

- --description-path: Path to the folder containing text descriptions (default: ./output).
- --output-path: Path to save the generated music (default: ./output).
- --model: MusicGen model to use (options: musicgen-small, musicgen-medium, musicgen-large).
- --duration: Length of the generated music in seconds (default: 10).
- --device: Device to run the model on (cuda or cpu).