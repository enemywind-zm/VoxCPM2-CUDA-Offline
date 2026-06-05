<div align="center">
  <img src="assets/voxcpm_logo.png" alt="VoxCPM Logo" width="35%">
  <h1>VoxCPM2: Pure Offline Text-to-Speech Engine</h1>
</div>

**VoxCPM2** is a highly realistic multilingual Text-to-Speech (TTS) model. This repository has been heavily customized and stripped down to serve as a **Pure Offline Inference Engine** with an interactive WebUI.

All training, LoRA, and network-dependent features have been removed to guarantee maximum stability, minimum dependencies, and full support for completely isolated environments.

## ✨ Key Features
- **Pure Offline Mode**: All models (VoxCPM2, SenseVoiceSmall, ZipEnhancer) are loaded directly from the local `./model/` directory. Zero runtime network requests.
- **Zero Redundancy**: Removed all training scripts, legacy model compatibility logic, and unused dependencies.
- **WebUI Driven**: Simple, out-of-the-box Gradio interface for quick Voice Cloning and Text-to-Speech generation.
- **Linux/CUDA Exclusive**: Stripped of macOS (MPS) and CPU fallbacks to ensure maximum performance purely on NVIDIA GPUs.

## 🚀 Installation & Setup

We recommend using `uv` to manage the virtual environment and dependencies for blazing-fast installations.

```bash
# 1. Create a virtual environment
uv venv .venv
source .venv/bin/activate

# 2. Install minimal dependencies from the locked file
uv sync
```

*Note: You MUST place your downloaded model weights into the `./model/` directory before running the engine. The required paths are:*
- `./model/OpenBMB/VoxCPM2`
- `./model/iic/SenseVoiceSmall`
- `./model/iic/speech_zipenhancer_ans_multiloss_16k_base`

## 🎮 Running the WebUI

To launch the graphical interface locally:

```bash
python app.py
```
*The UI will run on `http://0.0.0.0:8808`.*

## ⚙️ Core Parameters

When generating audio in the WebUI or via the core engine, you can control the following generation parameters:

- **CFG (Classifier-Free Guidance)**: Defaults to `1.5`. Controls how strongly the generation should follow the prompt/reference audio. Higher values enforce stricter cloning but may distort voice quality.
- **DiT Steps**: Defaults to `10`. The number of denoising diffusion steps. Fewer steps are faster, while more steps (up to 20-30) can yield slightly better acoustic details.
- **Reference Audio**: Used for voice cloning (few-shot). Provide a high-quality 3-5 second pure vocal `.wav` file as a reference.
- **Denoise Reference Audio**: Uses `ZipEnhancer` to automatically strip background noise from the reference audio before cloning.
- **Text Normalization**: Converts numbers and symbols into spoken words (e.g. `123` -> `one hundred and twenty three`) using `wetext`.

## 📜 License
The code in this repository is licensed under the Apache 2.0 License. The model weights are licensed separately by ModelBest (see official repository for details).
