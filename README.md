# Job Shielder

**Enterprise Recruitment Verification & Fraud Analytics Platform**

Job Shielder is a machine learning-powered platform designed to verify job postings and mitigate recruitment fraud risks in real-time. By leveraging advanced classification engines and behavioral threat indicators, the system helps enterprises and job seekers distinguish between legitimate opportunities and fraudulent schemes.

> **Disclaimer:** Job Shielder is a decision-support tool. Its outputs are indicators for further review, not definitive proof of legitimacy or fraud. Users should always verify employers through trusted, independent sources.

[![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)

---

## 📊 Dataset

The model is trained on the **EMSCAD benchmark dataset**, comprising **17,880 records**:

| Class | Count | Distribution |
| :--- | :--- | :--- |
| **Legitimate** | 17,014 | 95.16% |
| **Fraudulent** | 866 | 4.84% |
| **Total** | **17,880** | **100%** |

*Note: These figures represent the dataset's class distribution, not model accuracy metrics.*

---

## ⚙️ Classification Engine

Job Shielder utilizes a **Linear SVM** classifier paired with a **TF-IDF metadata pipeline** to evaluate textual patterns and structural anomalies in job postings.

### Core Features
*   **Real-Time URL Assessment:** Automatically parses job posting URLs to evaluate security risks.
*   **Manual Text Entry:** Analyze raw job description text for immediate threat scoring.
*   **Scan History Audit:** Maintain a transparent log of all verification attempts.
*   **Threat Reference Library:** A curated database of enterprise red flags and behavioral indicators.

---

## 🚩 Threat Intelligence & Red Flags

The system correlates historical data with known fraud patterns, including:

### Structural & Metadata Indicators
*   **Omission of Corporate Branding:** Absence of verified logos or organizational markers.
*   **Lack of Qualification Filters:** Elimination of standard multi-stage screening protocols.
*   **Broad Geographic Scope:** Undefined remote parameters targeted at scaling victim reach.

### Lexical & Financial Indicators
*   **Unverified Compensation Structures:** Outlier salary ranges promised for low-barrier tasks.
*   **Early Transactional Demands:** References regarding equipment procurement fees or payment clearing.
*   **Ambiguous Entity Profiles:** Generic descriptive terminology lacking formal registration details.

---

## 🛠 Tech Stack

*   **Core Language:** Python
*   **Machine Learning:** Linear SVM, TF-IDF Vectorization (Scikit-Learn)
*   **Frontend:** Streamlit
*   **Version Control:** Git / GitHub

---

## 🚀 Getting Started

### Prerequisites
*   Python 3.8+
*   pip

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/sidra-sultan/Job_Shielder.git
   cd Job_Shielder


2.**Install dependencies:**


```bash
pip install -r requirements.txt

```

3.**Run the application:**


```bash
streamlit run app.py
```
---
**Access the dashboard:**
 Open your browser and navigate to http://localhost:8501.

**👥 Team Developed by:**

* Sidra Sultan
* Noor Ul Ain Salamat Khan
* Sadaf Ahmed
* Javeria Noman
* Rida Baig

**🔗 Links**
Live Demo: https://sidra-sultan-job-shielder-app-qgf0vh.streamlit.app/

**📜 License**
This project is developed for academic and demonstration purposes. Feel free to fork, adapt, and build upon it for learning and research. Contact the maintainers for commercial use or redistribution.

**🤝 Contributing**
Contributions, issues, and feature requests are welcome! Feel free to check the issues page.

Fork the Project

Create your Feature Branch (git checkout -b feature/AmazingFeature)

Commit your Changes (git commit -m 'Add some AmazingFeature')

Push to the Branch (git push origin feature/AmazingFeature)

Open a Pull Request
