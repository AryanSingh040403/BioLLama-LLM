# BioLLama LLM: Medical Reasoning Adapter

![License](https://img.shields.io/badge/License-Apache%202.0-blue)
![HuggingFace](https://img.shields.io/badge/🤗-HuggingFace-yellow)
![Model](https://img.shields.io/badge/View%20Model-blue)
![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PEFT](https://img.shields.io/badge/PEFT-green)
![LoRA](https://img.shields.io/badge/LoRA-brightgreen)

## Abstract
BioLLama LLM is a specialized biomedical language model designed for high-accuracy medical question answering and clinical reasoning. Built upon the Llama 3.2 1B architecture, this model utilizes Parameter-Efficient Fine-Tuning (PEFT) via Low-Rank Adaptation (LoRA) to achieve domain adaptation while maintaining computational efficiency.

The model is fine-tuned on the MedMCQA dataset and optimized for Chain-of-Thought (CoT) reasoning, making it suitable for downstream clinical support tasks and educational assessment.

## Performance Benchmark

| Metric               | Score  | Context                                                         |
|----------------------|--------|-----------------------------------------------------------------|
| NEET PG 2024 (Subset) | 72.7%  | Zero-shot accuracy on select medical entrance exam questions.   |
