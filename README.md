# English to Urdu Legal Translation System 🇬🇧➡️🇵🇰

## 📌 Project Overview
Short introduction about the project.

This project is an English-to-Urdu Neural Machine Translation system
designed for legal and court-related text. The system was trained on
20,000 English-Urdu legal translation samples.

---

## 🎯 Objectives
- Translate English legal text into Urdu
- Compare different translation models
- Evaluate model performance using BLEU score and Loss
- Build an easy-to-use translation interface
- Deploy the application online

---

## 📊 Dataset
- Domain: Legal / Court-related text
- Total Samples: 20,000
- Languages: English → Urdu
- Dataset Type: Parallel Translation Dataset

---

## 🤖 Models Used

### 1. mBART
- Fine-tuned for English-to-Urdu translation

### 2. NLLB-200
- Fine-tuned for multilingual translation

### 3. Transformer (Base)
- Trained as a baseline Transformer model

---

## 📈 Model Performance

| Model | Validation Loss | BLEU Score |
|------|----------------:|-----------:|
| mBART | 0.4551 | **50.39** |
| NLLB-200 | **0.4178** | 48.34 |
| Transformer (Base) | 1.0898 | 25.13 |

### 🏆 Best Performing Model
mBART achieved the highest BLEU score of **50.39**, making it the best
performing model in terms of translation quality among the three models.

---

## 🛠️ Technologies Used

- Python
- PyTorch
- Hugging Face Transformers
- mBART
- NLLB-200
- Transformer
- Gradio
- Google Colab
- Hugging Face Spaces

---

## 🔄 Project Workflow

1. Collect legal translation dataset
2. Data preprocessing
3. Tokenization
4. Train/validation split
5. Model training
6. Model evaluation
7. Compare BLEU scores
8. Save trained models
9. Build Gradio interface
10. Deploy on Hugging Face Spaces

---

## 🖥️ Gradio Interface

Add screenshots of your application here.

![Gradio Interface](<img width="635" height="441" alt="nllb model" src="https://github.com/user-attachments/assets/bc1091b8-6c4b-4afb-8020-5f86c184d063" />
)

---

## 🚀 Live Demo

Try the deployed application:

👉 [Hugging Face Space](https://huggingface.co/spaces/areebanaz/english-urdu-legal-translation)

---

## 📂 Project Structure

```text
English-Urdu-Legal-Translation/
│
├── README.md
├── requirements.txt
├── app.py
├── notebook/
│   └── translation_project.ipynb
├── screenshots/
│   ├── gradio-ui.png
│   ├── mbart-training.png
│   ├── nllb-training.png
│   └── transformer-training.png
└── .gitignore
