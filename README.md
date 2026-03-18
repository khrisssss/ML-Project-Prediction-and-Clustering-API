# ML Prediction & Clustering Project

This project analyzes a dataset with columns **A**, **B**, and **Y**, builds a machine learning model to predict **Y**, performs clustering on **A** and **B**, and exposes everything through a FastAPI web service.

## Project Goal
- Explore and understand the data
- Answer 3 key prediction questions
- Build and compare models to predict **Y** from **A** and **B**
- Perform clustering to discover 3 natural groups
- Create a REST API for predictions and cluster assignment
- Provide an automated test script

## Project Structure
ML_Prediction_Clustering_Project/
├── Data/data.csv                 ← Original dataset
├── analysis_notebook.ipynb       ← Main analysis + model training
├── API.py                        ← FastAPI web service
├── test.py                       ← Automated testing script
├── best_model.pkl                ← Saved prediction model
├── scaler.pkl                    ← Scaler for clustering
├── kmeans_model.pkl              ← Clustering model
├── requirements.txt              ← Python dependencies
├── README.md                     ← This file
└── test_results.csv              ← Results after running test.py


## How to Run the Project (Step by Step)

### 1. Clone or Download the Project
```bash
git clone <your-repo-link>
cd ML_Prediction_Clustering_Project
```

### 2. Install Dependencies
``` Bash
pip install -r requirements.txt
```

### 3. Run the Analysis
Open the Jupyter notebook and rul all

### 4. Start the API Server
Open a terminal and run:
```bash
uvicorn API:app --host 127.0.0.1 --port 5000 --reload
```
Keep this terminal open

### 5. Run the Test Script
Open another terminal and run:
``` bash 
python test.py
```
This will create/update test_results.csv with predictions and cluster classes.

### 6. View the API Documentation
While the server is running, open your browser and go to:
"" http://127.0.0.1:5000/docs ""


# What Each File Does

analysis_notebook.ipynb → Full data analysis, model comparison, clustering, and saving models
API.py → Web API with two endpoints:
POST /predict → Predict Y from A and B
GET /classe → Get the cluster class for given A, B, Y

test.py → Automatically tests both API routes and saves results to CSV
best_model.pkl → The best trained model for predicting Y
scaler.pkl + kmeans_model.pkl → Used for clustering