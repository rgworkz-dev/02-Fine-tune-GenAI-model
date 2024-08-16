# Fine-Tune a Generative AI Model for Dialogue Summarization

This repository contains a Jupyter notebook that demonstrates how to fine-tune an existing large language model (LLM) from Hugging Face for enhanced dialogue summarization. The notebook focuses on the [FLAN-T5](https://huggingface.co/docs/transformers/model_doc/flan-t5) model, which is an instruction-tuned model capable of summarizing text.

## Overview

The notebook is divided into the following sections:

1. **Load Required Dependencies, Dataset, and LLM**
   - Load the necessary libraries and dependencies.
   - Load a dialogue-summary dataset and the FLAN-T5 model.
   - Test the model with zero-shot inference.

2. **Perform Full Fine-Tuning**
   - Preprocess the dialogue-summary dataset.
   - Fine-tune the FLAN-T5 model with the preprocessed dataset.
   - Evaluate the model qualitatively through human evaluation.
   - Evaluate the model quantitatively using the ROUGE metric.

3. **Perform Parameter Efficient Fine-Tuning (PEFT)**
   - Set up the PEFT/LoRA model for fine-tuning.
   - Train the PEFT adapter.
   - Evaluate the model qualitatively and quantitatively.

## Requirements

- Python 3.7 or above
- Jupyter Notebook
- Hugging Face Transformers
- Datasets
- PEFT (Parameter-Efficient Fine-Tuning)

You can install the required dependencies with the following command:

```bash
pip install -r requirements.txt
```

## How to Run

1. Clone this repository:
   ```bash
   git clone https://github.com/rgworkz-dev/02-Fine-tune-GenAI-model.git
   cd fine-tune-generative-ai-model
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Open the notebook:
   ```bash
   jupyter notebook Fine_tune_generative_ai_model.ipynb
   ```

4. Run the cells in the notebook sequentially to fine-tune the model and evaluate its performance.

## Results

The notebook provides both qualitative and quantitative evaluations of the fine-tuned model. The ROUGE metric is used to compare the performance of the fully fine-tuned model with the parameter-efficient fine-tuned model.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Hugging Face for providing the pre-trained FLAN-T5 model and tools.
- The open-source community for datasets and support.