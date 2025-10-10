# Real-Time Salience Capture

This is a tool for recording gameplay sessions for machine vision and reinforcement learning research. It intelligently saves high-quality video and data only when visually interesting or text-related events occur, while maintaining a low-quality baseline recording for context.

**THIS IS A TOOL FOR KEYLOGGING YOUR PERSONAL VIDEOGAME-USE SESSIONS FOR MACHINE VISION RESEARCH. BE ETHICAL, HONEST, AND KIND WITH USER INTERACTION DATA.** 

## INDEV:
major missing features:

X PCA decomposition (demi-implemented, hyperparameters aren't yet trustworthy)
- performance-max OCR capture (extremely strains cpu incremental-pca on use)
- rewrite to use obs-studio virtual camera API for various kinds of user data hygeine (avoid ever seeing emails or directories if user alt-tabs)
- audio capture (lol)
- reimplement low-level-of-detail low-bitrate 'baseline recording'
- data viewer
- rewrite to something like zeromq + trio so adding new features is less likely to crash your computer ;)
- vc funding 

## Core Features

-   **Salience-Based Recording**: Uses Vision Transformer models (SigLIP, DOTS) to detect significant visual and OCR events, saving high-quality footage and data only around these moments.
-   **Rich Data Capture**: Logs user input (keyboard/mouse), saves OCR latent vectors for offline analysis, and creates a queryable event timeline.

## Setup and Installation

This project uses `uv` for fast and reliable dependency management.

1.  **Install `uv`**:
    If you don't have `uv` installed, follow the official instructions. For most systems:
    ```bash
    curl -LsSf https://astral.sh/uv/install.sh | sh
    ```

2.  **Create a Virtual Environment**:
    From the project's root directory, create and activate a virtual environment:
    ```bash
    # Create the virtual environment in a .venv folder
    uv venv --seed

    # Activate it (on Linux/macOS)
    source .venv/bin/activate
    # On Windows (Powershell):
    # .venv\Scripts\Activate.ps1
    ```

3.  **Download Models**:
    The required neural network models are not included in the repository. Run the provided script to download them:
    ```bash
    python download_models.py
    ```

## Usage

1.  **Configure the Capture**:
    Open the `main.py` (or a `config.json` file) and set the `RECORDING_WINDOW_NAME` variable to the exact title of the game window you want to capture.

2.  **Run the Application**:
    ```bash
    python recording_stuff.py
    ```

3.  **Stop the Capture**:
    Press `Ctrl+C` in the terminal where the script is running. The application will perform a clean shutdown, finalizing and saving all data to the specified output path.