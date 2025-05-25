# PromptWeaver - AI Model Training and Workflow Management System

PromptWeaver is a comprehensive system for training, managing, and deploying generative AI models and ComfyUI workflows. This project demonstrates expertise in generative AI, model training, and workflow management.

## Features

- LoRA model training and fine-tuning
- ComfyUI workflow management and export
- RESTful API for model and workflow operations
- Training data management and preparation
- Model checkpoint management
- Workflow visualization and documentation

## Project Structure

```
PromptWeaver/
├── api/                    # RESTful API implementation
├── core/                   # Core functionality
│   ├── training/          # Model training modules
│   ├── workflows/         # ComfyUI workflow management
│   └── utils/             # Utility functions
├── data/                  # Training data management
├── models/                # Model checkpoints and LoRAs
├── workflows/             # ComfyUI workflow definitions
├── tests/                 # Unit and integration tests
└── docs/                  # Documentation
```

## Requirements

- Python 3.8+
- PyTorch
- ComfyUI
- FastAPI
- TensorFlow
- Other dependencies listed in requirements.txt

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/PromptWeaver.git
cd PromptWeaver
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Start the API server:
```bash
python api/main.py
```

2. Access the API documentation at `http://localhost:8000/docs`

3. Use the provided scripts for model training and workflow management:
```bash
python core/training/train_lora.py --config configs/training_config.yaml
```

## Documentation

Detailed documentation is available in the `docs/` directory:
- API Documentation
- Training Guide
- Workflow Management Guide
- Model Architecture Documentation

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

MIT License

## Contact

For questions and support, please open an issue in the repository.