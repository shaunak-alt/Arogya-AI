Fine-Tuning Gemma for Patient Dialogue 🩺
This project is a complete pipeline for fine-tuning the google/gemma-2b language model to simulate patient-doctor conversations. It leverages parameter-efficient fine-tuning (PEFT) with LoRA to create a specialized chatbot on a large corpus of medical Q&A data, optimized for consumer hardware or free cloud GPUs.

This system was built to address the need for realistic, AI-generated patient dialogue for applications like training medical students or augmenting healthcare datasets.

Key Features
Specialized Dialogue: Generates coherent, context-aware responses from a patient's perspective.

High-Performance Model: Built on Google's lightweight and powerful gemma-2b architecture.

Efficient Fine-Tuning: Uses LoRA and the Hugging Face PEFT library to fine-tune quickly with minimal VRAM.

Large-Scale Dataset: Combines the HealthCareMagic-100k and iCliniq datasets for robust training.

Reproducible Pipeline: A single script handles everything from data processing to training and testing.

Tech Stack
Backend: Python

ML Frameworks: PyTorch, Hugging Face (transformers, datasets, peft, accelerate)

Model: google/gemma-2b

Setup and Installation
Follow these steps to get the project running locally.

Prerequisites
Python 3.9+

An NVIDIA GPU with CUDA (highly recommended for training)

Git

1. Clone the Repository
git clone [https://github.com/your-username/your-repo-name.git](https://github.com/your-username/your-repo-name.git)
cd your-repo-name

2. Set Up the Environment
Create and activate a Python virtual environment.

# Create the virtual environment
python -m venv venv

# Activate it (on Linux/macOS)
source venv/bin/activate

# On Windows PowerShell
# .\venv\Scripts\Activate.ps1

3. Install Dependencies
Install all required packages.

pip install torch transformers datasets peft accelerate bitsandbytes

4. Download the Datasets
Create a folder named dataset in the project root and place your JSON dataset files inside it.

your-repo-name/
├── dataset/
│   ├── HealthCareMagic-100k.json
│   └── iCliniq.json
└── main.py

How to Run
First, log in to your Hugging Face account via the command line.

huggingface-cli login

Then, start the fine-tuning script.

python main.py

Model Usage
After the training process is complete, the script will automatically run a test inference pipeline.

Example Prompt
The test_prompt is defined at the end of main.py:

test_prompt = "### Instruction:\nYou are a patient speaking to a doctor. Describe your symptoms.\n\n### Input:\nHello, what seems to be the problem today?"

Example Response
The model will generate a response attempting to simulate a patient's answer.

### Output:
For the past few days, I've had a really sharp, throbbing headache right behind my eyes. It gets worse when I look at a screen, and I've been feeling a bit dizzy and nauseous too. I haven't taken any medication for it yet because I wanted to check with you first.

Disclaimer: This project is for educational and research purposes only. The generated output is not a substitute for professional medical advice, diagnosis, or treatment.
