import streamlit as st
import joblib
import numpy as np


# --- Load Models ---
models = {
    "Random Forest Classifier": joblib.load("models/random_forest_titanic.pkl"),
    "XGBoost Classifier": joblib.load("models/xgboost_titanic_model.pkl")
}

# --- Streamlit UI ---
st.set_page_config(page_title="🚢 Titanic Survivors Classifier", layout="centered")

st.title("🚢 Titanic Survivors Classifier")
st.markdown("Predict whether a passenger would have **survived the Titanic** disaster using trained ML (Random Forest) algorithms")


# --- Model Selection ---
model_choice = st.selectbox("Select Model", list(models.keys()))

# --- Display model configuration ---
st.subheader("🔧 Model Configuration")

model = models[model_choice]

with st.expander("🔍 View Model Parameters"):
    # Display key hyperparameters based on model type
    if model_choice == "Random Forest Classifier":
        st.json({
            "n_estimators": model.n_estimators,
            "max_depth": model.max_depth,
            "max_features": model.max_features,
            "random_state": model.random_state
        })
    elif model_choice == "XGBoost Classifier":
        st.json({
            "n_estimators": model.n_estimators,
            "learning_rate": model.learning_rate,
            "max_depth": model.max_depth,
            "scale_pos_weight": model.scale_pos_weight,
            "random_state": model.random_state
        })


st.write("Provide passenger details below:")

# --- Input Features ---
col1, col2 = st.columns(2)
with col1:
    pclass = st.selectbox("Ticket Class (1 = 1st, 2 = 2nd, 3 = 3rd)", [1, 2, 3])
    sex = st.selectbox("Sex", ["Male", "Female"])
    age = st.slider("Age", 1, 80, 29)
with col2:
    sibsp = st.number_input("Number of Siblings/Spouses Aboard", 0, 10, 1)
    parch = st.number_input("Number of Parents/Children Aboard", 0, 10, 0)
    fare = st.number_input("Ticket Fare (£)", 0.0, 600.0, 32.2)

# --- Preprocess Input ---
sex_value = 1 if sex == "Female" else 0
input_data = np.array([[pclass, sex_value, age, sibsp, parch, fare]])


# --- Prediction ---
if st.button("Predict Survival"):
    model = models[model_choice]
    prediction = model.predict(input_data)[0]
    proba = model.predict_proba(input_data)[0][1]

    if prediction == 1:
        st.success(f"🧍‍♀️ Passenger would have **SURVIVED** ({proba*100:.1f}% probability).")
    else:
        st.error(f"💀 Passenger would have **DIED** ({(1-proba)*100:.1f}% probability).")


# --- Educational Section ---
st.markdown("---")
st.header("📘 Educational Section: How the Model Works")

st.markdown("""
 ------- 
##  Data Preprocessing & Model Training
            
### 🧩 Exploratory Data Analysis (EDA)
EDA helped us understand the main factors influencing survival on the Titanic.  
We started by examining distributions, missing values, and correlations between variables.

**Steps performed:**
1. **Missing Values:**  
   - `Age`, `Fare`, and `Embarked` had missing entries.  
   - Replaced using **median** (for numeric) and **mode** (for categorical).

2. **Dropped Columns:**  
   - `Cabin`, `Ticket`, and `Name` were removed (too many unique or missing values).  
   - `PassengerId` was dropped as it has no predictive value.

3. **Encoding Categorical Data:**  
   - Converted text to numbers:  
     - `Sex` → 0 (Male), 1 (Female)  
     - `Embarked` → Encoded using `LabelEncoder`.

4. **Feature Selection:**  
   - Retained: `Pclass`, `Sex`, `Age`, `SibSp`, `Parch`, `Fare`, `Embarked`.

---

### ⚙️ Model Training Workflow
After cleaning and encoding, the dataset was split into:
- **80% training**, **20% testing** (stratified by survival).

These models predict survival based on key features:
- **Class** (social/economic status)
- **Sex** (gender)
- **Age**
- **Siblings/Spouses aboard**
- **Parents/Children aboard**
- **Fare paid**

            
#### We then trained two ensemble models: Random Forest and XGBoost.
- XGBoost **learns sequentially** to reduce errors step by step (boosting).  
- Random Forest **trains many independent trees** and averages results (bagging).  
- Both are strong ensemble models, but **XGBoost tends to perform better** after tuning.(not in this case)     

### Model Comparison

| Model | Accuracy | Recall (Survived) | F1-score | Notes |
|-------|-----------|--------------------|-----------|--------|
| Random Forest | 0.81 | 0.70 | 0.79 | Stable, balanced baseline |
| XGBoost | 0.81 | **0.74** | **0.80** | Better recall and generalization |

XGBoost was trained using the same features as the Random Forest model.  
It applies **boosting**, meaning each new tree learns from the errors of the previous one.

Despite tuning, XGBoost achieved **comparable but not superior performance**, likely due to the **small dataset size** and **low feature complexity**.
            
###  Key Insights
- **Women and first-class passengers** had the highest survival rates.
- **Fare and class** are strong predictors of survival likelihood.
- Both models perform around **81% accuracy** with stable results across folds.
""")

if __name__ == "__main__":
    st.set_page_config(page_title="🚢 Titanic Classifier", layout="centered")