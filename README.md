# 📄 Resume-JD Deep Neural Network Match Scorer

A Deep Learning-based project that predicts how well a candidate's resume matches a job description. The model classifies each Resume–Job Description pair into one of three categories:

- 🔴 Weak Match
- 🟡 Medium Match
- 🟢 Strong Match

Unlike traditional keyword matching systems, this project uses a **Siamese BiLSTM Deep Neural Network** to learn semantic relationships between resumes and job descriptions, resulting in more intelligent and accurate matching.

---

# 🚀 Features

- Deep Neural Network-based Resume Screening
- Siamese BiLSTM Architecture
- Resume & Job Description Semantic Matching
- Three-Class Prediction (Weak, Medium, Strong)
- Skill Overlap Detection
- Missing Skill Recommendations
- Model Evaluation with Accuracy, Precision, Recall & F1 Score
- Confusion Matrix Visualization
- Training History Graph
- Prediction Report Generation
- Explainable AI using Token Importance

---

# 🛠 Tech Stack

- Python
- TensorFlow / Keras
- NumPy
- Pandas
- Scikit-learn
- Matplotlib

---

# 📂 Project Structure

```
resume_jd_dnn_project/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── models/
│   ├── resume_jd_match_model.keras
│   ├── tokenizer.pkl
│   └── model_config.json
│
├── outputs/
│   ├── confusion_matrix.png
│   ├── evaluation_metrics.json
│   ├── prediction_result.md
│   └── training_history.png
│
├── src/
│   ├── preprocessing.py
│   ├── model.py
│   ├── train.py
│   ├── evaluate.py
│   ├── predict.py
│   └── explain.py
│
├── tests/
│
├── Resume_JD_Deep_Neural_Network.ipynb
├── requirements.txt
└── README.md
```

---

# 🧠 Model Architecture

The project uses a **Siamese Neural Network** with a shared **BiLSTM encoder**.

```
Resume
   │
Embedding
   │
BiLSTM
   │
Global Max Pooling
   │
 Resume Vector
                 \
                  \
                   Feature Comparison
                  /
                 /
JD
 │
Embedding
 │
BiLSTM
 │
Global Max Pooling
 │
JD Vector

↓

Dense Layers

↓

Softmax Classifier

↓

Weak / Medium / Strong Match
```

The model compares both document embeddings using:

- Concatenation
- Absolute Difference
- Element-wise Multiplication

These features help the network understand semantic similarity between resumes and job descriptions.

---

# 📊 Dataset

The dataset contains:

- Job Descriptions
- Matching Resumes
- Non-Matching Resumes
- Filtered Information

During preprocessing, each sample is converted into:

| Resume Type | Label |
|-------------|-------|
| Matching Resume | Strong (2) |
| Modified Resume | Medium (1) |
| Unmatched Resume | Weak (0) |

This creates a balanced three-class classification dataset.

---

# ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/yourusername/resume-jd-dnn-project.git

cd resume-jd-dnn-project
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# ▶️ Train the Model

```bash
python src/train.py
```

The training script will:

- Load the dataset
- Tokenize text
- Train the Siamese BiLSTM model
- Save the trained model
- Save tokenizer
- Generate training history plots

---

# 📈 Evaluate the Model

```bash
python src/evaluate.py
```

Evaluation includes:

- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix

The results are stored inside the **outputs/** folder.

---

# 🔍 Predict Resume Match

```bash
python src/predict.py
```

Example:

```text
Resume:
Python Developer with TensorFlow, Docker, FastAPI

Job Description:
Looking for AI Engineer skilled in Python, FastAPI, TensorFlow, Docker

Prediction:

Strong Match
Confidence: 95%

Common Skills:
Python
TensorFlow
Docker
FastAPI

Missing Skills:
None
```

---

# 📖 Explain Predictions

The project also provides an explainability module.

```bash
python src/explain.py
```

It highlights important words or phrases that influenced the model's prediction, making the system more transparent and interpretable.

---

# 📊 Outputs

After execution, the project generates:

- Trained Model (.keras)
- Tokenizer (.pkl)
- Model Configuration
- Training History Plot
- Confusion Matrix
- Evaluation Metrics (JSON)
- Prediction Report (Markdown)

---

# 📌 Applications

- AI Resume Screening
- Recruitment Automation
- HR Analytics
- Applicant Tracking Systems (ATS)
- Candidate Ranking
- Resume Recommendation Systems

---

# 🔮 Future Improvements

- Transformer-based models (BERT, RoBERTa)
- Attention Mechanism
- PDF Resume Parsing
- OCR Support
- Skill Extraction using Named Entity Recognition
- Multi-language Resume Matching
- Web-based Deployment using Flask or FastAPI

---

# 👨‍💻 Author

**Ashutosh Gupta**

B.Tech (Electronics & Communication Engineering)

Birla Institute of Technology, Mesra – Patna Campus

GitHub: https://github.com/Ashu67-debug

---

# ⭐ If you found this project useful, consider giving it a Star on GitHub!
