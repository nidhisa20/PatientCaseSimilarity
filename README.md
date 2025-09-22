# deploy-ml-model-flask
 deploy your ml model using flask

 # 🏥 Patient Case Similarity
 
A **Patient Case Similarity System** that identifies and retrieves similar patient cases from historical medical data. This tool helps healthcare professionals make better decisions by comparing new patient reports with past cases.

---

## 🏆 Project Overview

Developed a **machine learning system** for early diagnosis and personalized treatment by identifying comparable historical patient cases.  

- Implemented **advanced preprocessing** to handle categorical and numerical clinical data.  
- Built a **machine learning model** to predict diseases and find similar past cases.  
- Deployed using **Flask**, providing a **user-friendly interface** for symptom input and disease forecasting.  
- Enables healthcare professionals to access **preventive strategies** and make informed decisions.

---

## 🛠 Features & Predictions

### Features
- Patient case similarity matching  
- Symptom-based search input  
- Preprocessing & scaling of data  
- SVM model for similarity calculation  
- Flask web interface for easy input/output  
- Reusable serialized models and encoders  

### What It Predicts
- **Similarity Score**: How similar a new patient case is to past cases  
- **Similar Past Cases**: Returns a list of comparable historical cases  
- **Potential Diagnosis Aid**: Supports healthcare professionals in treatment decisions  

---

## 📁 Project Structure

```plaintext
PatientCaseSimilarity/
├── app.py                      # Main Flask application
├── basics.py                   # Helper functions
├── iris.py                     # Example/testing script
├── label_encoder.pkl           # Serialized label encoder
├── scaler.pkl                  # Serialized scaler
├── sv_model.pkl                # Pre-trained SVM model
├── Disease_symptom_and_patient_profile_dataset.csv  # Dataset
├── templates/                  # HTML templates
│   └── index.html              # Main page
├── static/                     # CSS/JS and other static files
├── requirements.txt            # Python dependencies
├── Procfile                    # Deployment config
├── README.md                   # Documentation
└── LICENSE                     # MIT License

```
## 🖼 Screenshots

### Home Page / Input Form

<img width="1654" height="883" alt="Screenshot 2025-09-23 014318" src="https://github.com/user-attachments/assets/963f05f4-6a16-462f-8d19-d4a8851ca5e3" />


### Similar Cases Result

<img width="1511" height="896" alt="Screenshot 2025-09-23 014255" src="https://github.com/user-attachments/assets/3d137d7d-215d-4c87-9bc4-29327d495655" />


## 🚀 How to Run Locally

1. **Clone the repository:**
```bash
git clone https://github.com/nidhisa20/PatientCaseSimilarity.git
cd PatientCaseSimilarity
```
2. Create a virtual environment:
   ```python -m venv venv```
3. Activate the virtual environment:
   ```venv\Scripts\activate```
4. Install dependencies:
    ```pip install -r requirements.txt ```
5. Run the Flask app:
    ``` python app.py```
6. Open in your browser:
  ``` http://127.0.0.1:5000/```

## ⚡ Future Enhancements

- Use advanced **NLP techniques** for better similarity scoring  
- Integrate with **real hospital databases** securely  
- Add **charts/visualizations** for similarity results  
- Include **user authentication** for healthcare professionals





 

