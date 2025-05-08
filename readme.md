# Flask-Server - Autonance


This is a **Flask-based backend** built for **Autonance**, focused purely on machine learning and AI processing.
## 🧩 Core Components

<table>
  <tr>
    <th>Module</th>
    <th>Description</th>
    <th>Technologies</th>
  </tr>
  <tr>
    <td><strong>Mandate Approval Prediction</strong></td>
    <td>Predicts if a mandate should be approved or rejected.</td>
    <td>Flask, scikit-learn (DecisionTree, RandomForest)</td>
  </tr>
  <tr>
    <td><strong>Loan Default Predictor</strong></td>
    <td>Predicts if a user is likely to default on a loan.</td>
    <td>TensorFlow (ANN), Flask</td>
  </tr>
  <tr>
    <td><strong>KYC Verification</strong></td>
    <td>Extracts Aadhaar and PAN details via OCR.</td>
    <td>Tesseract, OpenCV</td>
  </tr>
  <tr>
    <td><strong>AI Chatbot</strong></td>
    <td>Answers user queries using internal documents.</td>
    <td>Mistral LLM, FAISS, Sentence-Transformers</td>
  </tr>
</table>

---

## 🧑‍💼 Roles & Responsibilities

<table>
  <tr>
    <th>Actor</th>
    <th>Responsibilities</th>
  </tr>
  <tr>
    <td><strong>User</strong></td>
    <td>Submits mandate requests, uploads KYC docs, interacts with chatbot.</td>
  </tr>
  <tr>
    <td><strong>Bank</strong></td>
    <td>Reviews predictions and takes final decisions on mandates and loans.</td>
  </tr>
  <tr>
    <td><strong>Compliance Team</strong></td>
    <td>Manages and verifies KYC data.</td>
  </tr>
</table>

---

## 🧩 Diagram 

<img src="https://res.cloudinary.com/dte1f5c5z/image/upload/fl_preserve_transparency/v1746695765/flask_diagram_zsngds.jpg?_s=public-apps" width="800">
