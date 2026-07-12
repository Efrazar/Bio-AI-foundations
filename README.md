# 🧬 Bio-AI Foundations & Neural Lab
## Dr. Efrain Zarazua-Arvizu, PhD | Molecular Biology & Deep Learning Practitioner
Welcome to my technical workspace. This repository serves as a "Digital Lab Notebook" documenting my transition from Molecular Biology to Applied Artificial Intelligence.
As a scientist, my goal is to leverage Deep Learning to solve complex problems in biotechnology, ranging from protein structure prediction to automated genomic analysis. This space tracks my journey from foundational theory to practical, fine-tuned Bio-AI agents.


### 🖥️ The Local AI Server (Hardware)
To maintain data privacy and allow for rapid iteration on proprietary biological datasets, I operate a custom local inference and training rig:\
*	Host: HP EliteBook 840 G5 (Intel i7 | 32GB DDR4 RAM)\
*	Accelerator (eGPU): NVIDIA RTX 2080 Ti (11GB VRAM) via Razer Core X (Thunderbolt 3)\
*	Environment: Ubuntu / WSL2 with PyTorch & CUDA optimization\
*	Focus: Efficient fine-tuning of distilled models (PEFT/QLoRA) and Protein Language Models (ESM-2).

### 📚 Learning Roadmap: Theory to Implementation
I am currently bridging the gap between biological theory and computational architecture through the following rigorous frameworks:
1. Architectural Foundations
*	Source: Inside Deep Learning by Edward Raff
*	Focus: Implementing neural architectures from scratch (Fully connected networks, CNNs, RNN, Autoencoders, GANs), understanding backpropagation, and manual gradient descent optimization.
*	Bio-Link: Translating signal processing in cells into mathematical weight distributions.
2. Visual & Intuitive Deep Learning
*	Source: Deep Learning: A Visual Approach by Andrew Glassner
*	Focus: Computer vision, CNNs, and the geometric intuition behind high-dimensional data.
*	Bio-Link: Automating microscopy analysis and identifying patterns in spatial transcriptomics.

### 🧪 Current Bio-AI Research Interests (active learning)
My work is evolving toward the practical application of Small Language Models (SLMs) and Agents in the wet-lab and dry-lab environments:
* Fine-Tuning for Bio-Domain: Adapting Llama 3.2 (3B) and Qwen 2.5 (7B) to interpret proprietary lab protocols and PubMed data.
*	Protein Folding & Design: Deploying ESM-2 locally for mutation effect prediction.
* AI Agents for Scientists: Building LangChain-based agents to automate BLAST searches and primer design.

### 📁 Repository Structure
Bio-AI-Foundations\
.
├── Deep Learning\
│   └── Deep_Learning.ipynb\
├── DL_utils_EZA\
│   ├── biggan_semantic_vectors.py\
│   ├── dl_utils.py\
│   └── __pycache__\
├── gradient descent\
│   ├── gradient_descent_example.py\
│   ├── GRADIENT_DESCENT_FUNCTIONS.py\
│   └── Gradient_descent_test_1.py\
├── hardware\
│   ├── export_report.py\
│   ├── hardware_validator.py\
│   ├── __pycache__\
│   └── system_report.svg\
├── HARDWARE.md\
├── Inside_Deep_Learning\
│   ├── 01_Mechanics_of_learning_exercises.ipynb\
│   ├── 02_Fully_Connected_Networks_exercises.ipynb\
│   ├── 02_Fully_Connected_Networks.ipynb\
│   ├── 03_CIFAR10_CNN_EZA.ipynb\
│   ├── 03_CNN_exercise_1.ipynb\
│   ├── 03_Convolutional_Neuronal_Networks.ipynb\
│   ├── 04_Recurrent_Neural_Networks.ipynb\
│   ├── 05_Modern_training_techniques.ipynb\
│   ├── 06_Common_design_building_blocks.ipynb\
│   ├── 07_Autoencoding_&_Self-supervision.ipynb\
│   ├── 08_Object_Detection.ipynb\
│   ├── 09_Generative_Adversarial_Networks.ipynb\
│   ├── checkpoints\
│   ├── checkpoints_wgan_gp\
│   ├── data\
│   ├── Foundational_Methods\
│   ├── idlmam.py\
│   ├── images_test\
│   ├── __pycache__\
│   └── PyTorch_training.ipynb\
└── README.md

### 🤝 Connect with Me
I am always interested in collaborating on Open Science and AI-driven drug discovery.
*	GitHub: Efrazar
*	Field: Molecular Biology / Bioinformatics / Deep Learning
