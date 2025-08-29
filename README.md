# CApLM: CAZyme Prediction with Protein Language Models

## ⚙️ Installation

1.  **Clone the Repository**
    ```bash
    https://github.com/lczong/CApLM.git
    cd CApLM
    ```

2.  **Set Up a Virtual Environment (Recommended)**
    ```bash
    conda create -n caplm
    conda activate caplm
    ```

3.  **Install Dependencies**
    ```
    tqdm
    numpy
    biopython
    torch
    transformers
    ```

## 📖 Usage

### Basic Prediction

This is the simplest way to run a prediction. The command is designed to work out-of-the-box. It will
- Using the default classification thresholds
- Automatically detecting devices (CPU or GPU)
- Downloading the required model weights from the [CApLM](https://huggingface.co/lczong/CApLM) repository on the Hugging Face Hub.

```bash
python src/predict.py --input example/example.fasta 
```