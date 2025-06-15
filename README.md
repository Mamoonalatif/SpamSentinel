# 📧 SPAM SENTINEL – AI-Powered Message Classifier
![image](https://github.com/user-attachments/assets/fcd62ade-ead3-40b7-a5c7-c2fd9fc27cca)
![image](https://github.com/user-attachments/assets/0a09f833-4400-47b0-bf92-eb4ea25b8f26)
![image](https://github.com/user-attachments/assets/62ba7d5c-1452-4356-8c5b-bcbc2a4bd4c9)
![image](https://github.com/user-attachments/assets/c37a4fac-051f-4f49-9219-715986ae3703)

A visually captivating, red-black themed **Streamlit web app** that detects spam messages using **Support Vector Machines (SVM)** with a **linear kernel**. The classifier is powered by a custom feature extractor and trained using fundamental principles of **Lagrange multipliers** in convex optimization.

---

## 🚀 Features

* 🔒 **Red-Black Security UI**: Professional dark-themed interface with background image overlay.
* ✉️ **Live Message Analysis**: Detect spam characteristics in any input text.
* 📊 **Real-time Metrics**: Message length, exclamation count, and spam keyword frequency.
* 🧠 **AI Classifier**: Linear SVM trained on handcrafted features.
* 📁 **Training Dataset Viewer**: See the labeled data used in training.
* 📉 **Probability Feedback**: Uses `predict_proba` for confidence-based classification.

---

## 📌 Classifier Architecture

### 🎯 Features Extracted

Each message is processed to extract a **3-dimensional feature vector**:

| Feature         | Description                                            |
| --------------- | ------------------------------------------------------ |
| `Length`        | Total characters in the message                        |
| `Exclamations`  | Number of `!` characters                               |
| `Spam Keywords` | Count of known spam terms (like "free", "offer", etc.) |

---

### 🔍 Linear SVM Formulation

We use a **hard-margin Linear SVM** as the base classifier. The optimization problem solved (conceptually) is:

#### **Primal Form:**

Minimize:

```
    ½ ||w||²
```

Subject to:

```
    yᵢ(w·xᵢ + b) ≥ 1   for all i
```

---

### ⚖ Lagrange Dual with Multipliers

We move to the **dual form** using Lagrange multipliers `αᵢ`, forming the **Lagrangian**:

```
L(w, b, α) = ½ ||w||² - Σ αᵢ [yᵢ (w·xᵢ + b) - 1]
```

We solve the dual problem:

Maximize:

```
    W(α) = Σ αᵢ - ½ Σ Σ αᵢ αⱼ yᵢ yⱼ (xᵢ · xⱼ)
```

Subject to:

```
    Σ αᵢ yᵢ = 0,
    αᵢ ≥ 0
```

In practice, this is efficiently solved using libraries like **scikit-learn**’s `SVC(kernel='linear', probability=True)`.

---

## 🛠 Tech Stack

| Tool               | Purpose                          |
| ------------------ | -------------------------------- |
| **Python**         | Core logic and model training    |
| **Streamlit**      | Web UI for interaction           |
| **scikit-learn**   | SVM training and prediction      |
| **Pandas & NumPy** | Data manipulation                |
| **Custom CSS**     | Modern red-black mailbox-like UI |

---

## 🧪 Training Data

10 labeled messages used for demonstration purposes:

* `1` = Spam (e.g., "FREE iPhone!!!")
* `0` = Safe (e.g., "Team meeting at 3pm")

More robust performance can be achieved by expanding this dataset.

---

## 📷 UI Preview

| Section        | Description                            |
| -------------- | -------------------------------------- |
| **Header**     | "Spam Sentinel" logo in bold red theme |
| **Input Box**  | User enters a suspicious message       |
| **Metrics**    | Shows extracted features in real-time  |
| **Result Box** | Classifies message as `SPAM` or `SAFE` |
| **Expander**   | Displays the training dataset          |

---

## ⚡ How to Run

```bash
pip install streamlit scikit-learn pandas numpy
streamlit run spam_sentinel.py
```

---
