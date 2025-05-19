# AI Health & Nutrition Assistant

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.31.0-red.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

A Streamlit-based web application delivering personalized diet plans, workout routines, nutrition advice, and food image analysis using advanced AI models. Powered by an agentic model with Gemini and a fine-tuned Flan-T5 model with LoRA, the app supports multimodal inputs (text, voice, image) for accessible health guidance.

[**Try the Demo on Streamlit Cloud**](https://your-app.streamlit.app)

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Demo Usage](#demo-usage)
- [Local Installation](#local-installation)
- [Model Architecture](#model-architecture)
- [Performance Metrics](#performance-metrics)
- [Future Work](#future-work)
- [Contributing](#contributing)
- [License](#license)

## Overview

The **AI Health & Nutrition Assistant** provides personalized health and nutrition guidance through a user-friendly Streamlit interface. Key components include:
- **Agentic Model**: Orchestrates tasks using Gemini for comprehensive health strategies.
- **Fine-Tuned Flan-T5**: Delivers precise nutrition advice with LoRA adaptation.
- **Multimodal Capabilities**: Processes text, voice (speech-to-text), and images (food analysis).

Deployed on Streamlit Cloud, the app offers instant access to its features, with source code available for local development and contribution.

## Features

- **Personalized Health Plans**: Tailored diet and workout plans based on user details (age, weight, height, activity level, dietary preferences, fitness goals).
- **Nutrition Queries**: Detailed meal suggestions with nutritional breakdowns (calories, carbs, protein, fat) via Flan-T5.
- **Food Image Analysis**: Identifies foods, estimates calories, and provides health assessments using Gemini’s vision capabilities.
- **Multimodal Inputs**: Supports text, voice (via `speech_recognition`), and image uploads.
- **Accessible Outputs**: Responses in Markdown, text-to-speech audio (`gTTS`), and JSON downloads.
- **Robust Sanitization**: Ensures input validity (<200 characters, no repetition).
- **Extensible Design**: Mocked `DuckDuckGoTools` for future web integration.

## Demo Usage

1. **Access the App**:
   - Visit  -  https://agenticspeechpy-t4cuuveszvy8rpgry2u9vj.streamlit.app/

2. **Navigate the UI**:
   - **Sidebar**: Enter user details (name, age, weight, height, activity level, dietary preferences, fitness goals).
   - **Tabs**:
     - **Health & Fitness Plan**: Generate diet and workout plans.
     - **Nutrition Query**: Input text or voice queries (e.g., “Suggest a low-carb breakfast”).
     - **Food Image Analysis**: Upload food images (JPG/PNG) for nutritional analysis.

3. **Example Inputs**:
   - **Health Plan**: Name: John Doe, Age: 30, Weight: 70kg, Height: 170cm, Activity: Moderate, Preference: Low Carb, Goal: Weight Loss.
   - **Nutrition Query**: “Suggest a low-carb breakfast” (text or voice).
   - **Food Image**: Upload an image of idli and sambar.

4. **Outputs**:
   - Markdown tables for plans and queries.
   - Audio playback for accessibility.
   - JSON downloads for portability.

**Note**: Voice input requires microphone access; ensure browser permissions are enabled.

## Local Installation

For developers or contributors who want to run the app locally or modify the code:

### Prerequisites
- Python 3.8+
- Git
- API key for Gemini (`google-generativeai`)

### Model Architecture
Agentic Model
Purpose: Orchestrates tasks in the Health & Fitness Plan tab.
Components:
Dietary Planner: Generates meal plans using Gemini.
Fitness Trainer: Creates workout plans.
Team Lead: Integrates plans into a holistic strategy.
Workflow: Input sanitization → Prompt construction → Gemini inference → Output cleaning → Markdown delivery.
Tools: Mocked DuckDuckGoTools for future web integration.
Flan-T5 Model
Purpose: Handles nutrition queries with precision.
Architecture:
Seq2seq transformer (flan-t5-base, ~250M parameters).
12-layer encoder and decoder with self-attention and feed-forward networks.
LoRA adaptation: ~3M trainable parameters (r=8, lora_alpha=32, lora_dropout=0.1, target_modules=["q", "v"]).
Workflow: Input sanitization → Prompt construction → Tokenization (AutoTokenizer) → Inference (beam search, num_beams=4) → Output cleaning → Multimodal delivery (UI, audio, JSON).
Training Data: Fine-tuned on merged_nutrition_dataset.json for 3 epochs.
Performance Metrics
Flan-T5 Fine-Tuning:
Training Loss: 3.5154 (Step 500)
Validation Loss: 2.7916 (Step 500)
Analysis: Lower validation loss indicates stable generalization, with no overfitting. Moderate loss values suggest potential for further training or dataset augmentation.
Agentic Model: Reliable text generation via Gemini, with consistent output quality across agents.
Future Work
Model Optimization: Extend Flan-T5 training, tune LoRA parameters (e.g., r=16), or adjust learning rate to reduce losses.
Real-Time Data: Integrate actual DuckDuckGo searches for up-to-date nutritional data.
Dataset Expansion: Augment merged_nutrition_dataset.json with diverse, culturally relevant examples (e.g., Indian cuisine).
Evaluation Metrics: Implement BLEU/ROUGE scores for nutrition query quality.
Feature Enhancements: Add user progress tracking (calorie logs, workout history) and mobile app support.
Scalability: Enhance cloud deployment with API integration (see xAI API).
Contributing
Contributions are welcome! To contribute:

Fork the repository.
Create a feature branch (git checkout -b feature/your-feature).
Commit changes (git commit -m "Add your feature").
Push to the branch (git push origin feature/your-feature).
Open a pull request.
Please follow the Code of Conduct and ensure tests pass.

License
This project is licensed under the MIT License. See the LICENSE file for details.
