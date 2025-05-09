# DeepSeek-Coder RAG Application

This application combines DeepSeek-Coder 33B model with RAG (Retrieval-Augmented Generation) to provide context-aware coding assistance. It includes both a Streamlit web UI and a Telegram bot interface.

## Features

- Run DeepSeek-Coder 33B locally with 4-bit/8-bit quantization
- Upload project ZIP files to create context for code-related questions
- Extract and analyze code from images/screenshots
- Query the model with or without project context
- Support for chunking large responses if they exceed token limits
- Two interfaces: Streamlit web UI and Telegram bot

## Requirements

- Python 3.8+
- CUDA-compatible GPU with 16+ GB VRAM (preferably 24+ GB) for optimal performance
- 32+ GB of system RAM
- Internet connection for model downloads
- Tesseract OCR (for code image processing)
  - On Ubuntu/Debian: `sudo apt-get install tesseract-ocr`
  - On MacOS: `brew install tesseract`
  - On Windows: Download from [GitHub](https://github.com/UB-Mannheim/tesseract/wiki)

## Installation

1. Clone the repository and navigate to it:

```bash
git clone https://github.com/yourusername/deepseek-rag-app.git
cd deepseek-rag-app
```

2. Create a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install the dependencies:

```bash
pip install -r requirements.txt
```

4. Set up environment variables by copying the example:

```bash
cp .env.example .env
```

5. Edit the `.env` file and fill in the required values.

## Usage

### Running the Streamlit Web Interface

```bash
python main.py streamlit
```

This will start the Streamlit server, typically accessible at `http://localhost:8501`.

### Running the Telegram Bot

1. First, obtain a Telegram Bot Token from the [BotFather](https://t.me/botfather).
2. Add the token to your `.env` file as `TELEGRAM_BOT_TOKEN=your_token_here`.
3. Run the bot:

```bash
python main.py telegram
```

## RunPod Deployment Guide

This guide explains how to deploy the application on RunPod for cost-effective yet powerful inference.

### Setting Up a RunPod Instance

1. **Create a RunPod account**:
   - Sign up at [RunPod.io](https://www.runpod.io/)

2. **Select the appropriate GPU**:
   - Recommended: RTX A6000 (48GB), RTX A5000 (24GB), or RTX 4090 (24GB)
   - For the best cost/performance ratio, choose RTX 4090
   - Minimum requirement: 16GB VRAM (RTX 3090/4080/A4000)

3. **Choose a template**:
   - Select "PyTorch 2.2" or "Runpod Pytorch" template
   - This provides a good foundation with CUDA drivers pre-installed

4. **Allocate disk space**:
   - Minimum 30GB for the model and project indices

### Deployment Steps

1. **Connect to your pod via SSH or use the web terminal**

2. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/deepseek-rag-app.git
   cd deepseek-rag-app
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**:
   ```bash
   cp .env.example .env
   # Edit the .env file with your values
   nano .env
   ```

5. **Expose the port (for Streamlit)**:
   - Go to your pod's settings in RunPod dashboard
   - Add a port forward: 8501 → 8501 (for Streamlit)

6. **Run the application with screen (to keep it running)**:
   ```bash
   # Install screen if not available
   apt-get update && apt-get install -y screen
   
   # Create a new screen session
   screen -S deepseek-app
   
   # Run the application
   python main.py streamlit  # or: python main.py telegram
   
   # Detach from screen with Ctrl+A, then D
   ```

7. **Access the web interface**:
   - Use the exposed URL from RunPod dashboard (usually like https://your-pod-id-8501.proxy.runpod.net)

### Cost Optimization Tips

1. **Use Spot Instances**:
   - RunPod's Spot instances are significantly cheaper (50-70% less than on-demand)
   - Suitable for development and testing
   - For production, consider on-demand for stability

2. **Optimize GPU memory usage**:
   - The app is configured to auto-detect GPU memory and use 4-bit or 8-bit quantization
   - This allows you to use smaller, cheaper GPUs without compromising much on quality

3. **Turn off when not in use**:
   - Pause or stop your pod when not actively using it
   - Use RunPod's scheduled autoscaling for recurring usage patterns

4. **Save your pod as a template**:
   - After setting everything up, save your pod as a custom template
   - This lets you quickly redeploy without reinstalling everything

5. **Utilize volume storage**:
   - Create a persistent volume for model storage
   - Attach this volume to new pods to avoid re-downloading models

## Project Structure

```
deepseek-rag-app/
├── app/
│   ├── streamlit_app.py   # Streamlit web interface
│   └── telegram_bot.py    # Telegram bot implementation
├── utils/
│   ├── model_loader.py    # Model loading utilities
│   ├── rag_utils.py       # RAG implementation for project context
│   └── chunking.py        # Response chunking utilities
├── data/
│   └── indices/           # Storage for project indices
├── models/                # Model weights cache
├── .env.example           # Example environment variables
├── main.py                # Main entry point
├── README.md              # This documentation
└── requirements.txt       # Python dependencies
```

## Limitations

- The application requires significant GPU memory to run efficiently
- Initial model loading can take time, especially on slower internet connections
- Very large projects may need to be split into multiple ZIPs for processing
- The model works best with code-related questions and may struggle with general queries

## License

This project is released under the MIT License.