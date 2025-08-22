# Diabetes Prediction System  
# Project Overview  
This project is a **machine learning-based system** that predicts the likelihood of diabetes using health-related data. 
It uses multiple classification algorithms including:  
- Logistic Regression  
- Support Vector Machine (SVM)  
- K-Nearest Neighbors (KNN)  
- Random Forest  

The dataset is preprocessed, visualized, and used to train models.
The system also accepts **user input** for real-time predictions.  

## Dataset  
- **Source:** `diabetes_prediction_dataset.csv` (100,000 rows × 9 columns)  
- **Features:**  
  - Gender  
  - Age  
  - Hypertension  
  - Heart Disease  
  - Smoking History  
  - BMI  
  - HbA1c Level  
  - Blood Glucose Level  
- **Target:** `diabetes` (0 = No, 1 = Yes)  

## Technologies Used  
- Python (NumPy, Pandas)  
- Scikit-learn (ML models, preprocessing)  
- Matplotlib & Seaborn (data visualization)  

## Exploratory Data Analysis (EDA)  
- Age and BMI distribution analysis  
- Histograms of numerical features  
- Pair plots & correlation heatmap  
- Bar plots for categorical variables (gender, smoking history, etc.)  

## Machine Learning Models  
| Model                | Accuracy |  
|----------------------|----------|  
| Logistic Regression  | 96.03%   |  
| Support Vector Machine (SVM) | 95.01%   |  
| K-Nearest Neighbors (KNN) | 95.29%   |  
| Random Forest        | 97.12%   |  

Random Forest performed best.  

## 🖥How to Run the Project  

## 1. Clone the Repository  
git clone https://github.com/your-username/diabetes-prediction-system.git
cd diabetes-prediction-system
## 2. Install Dependencies
pip install -r requirements.txt
## 3. Run the Notebook
Open Jupyter Notebook / Google Colab and run the cells to train and test models.
## 4. Make Predictions
Run the script to enter your health details and get predictions:
python predict.py
Example:
Age: 45
Gender (Male/Female/Other): Male
Hypertension (0/1): 0
Heart Disease (0/1): 0
Smoking History: current
BMI: 28.5
HbA1c Level: 6.2
Blood Glucose Level: 130
# Output:
Predictions:
Logistic Regression: 1
SVM: 1
KNN: 0
Random Forest: 1
## Results:
* Random Forest achieved the **highest accuracy (97.12%)**.
* Logistic Regression and KNN also performed well.
* Visualizations provided useful insights into health parameters affecting diabetes.
  
## Future Improvements:
* Hyperparameter tuning for better accuracy.
* Deploying the model as a **Flask/Django web app**.
* Adding explainability with **SHAP/LIME**.
