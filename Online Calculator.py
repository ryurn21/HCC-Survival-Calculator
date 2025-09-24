import streamlit as st
import pandas as pd
import joblib
import torch
import pycox.models.coxph as coxph
CoxPH = coxph.CoxPH
import torchtuples as tt
import matplotlib.pyplot as plt
import torch.nn as nn
import numpy as np

# Load model and preprocessor
model_path = '/Users/ryanlim/Desktop/University/IMGS/Cohort HCC Score/Model/cox_model.pt'
mapper_path = '/Users/ryanlim/Desktop/University/IMGS/Cohort HCC Score/Model/x_mapper.pkl'


# Define model architecture (must match training)
in_features = 13  # or use len(x_mapper.features) after loading
net = tt.practical.MLPVanilla(in_features, [48, 16], 1, batch_norm=True, dropout=0.3)
model = CoxPH(net)
torch.serialization.add_safe_globals([
    tt.practical.MLPVanilla,
    nn.Sequential,  # needed for internal architecture
    nn.Linear,
    nn.ReLU,
    nn.BatchNorm1d,
    nn.Dropout,
])
model.load_net(model_path, weights_only=False)

x_mapper = joblib.load(mapper_path)

# Extract feature names for input
feature_names = []
for feat_tuple in x_mapper.features:
    name = feat_tuple[0]
    if isinstance(name, list):
        feature_names.append(name[0])
    else:
        feature_names.append(name)

# Define which variables are binary and the mapping for 'newdefinition'
binary_vars = ['t2diabetes', 'listascites', 'HEwaitlist']

#ethnicity_options = {
#    "Asian": 0,
#    "Caucasian": 1
#}

newdefinition_options = {
    "MASLD": 1,
    "HCV": 2,
    "ALD": 3,
    "HBV": 4,
    "Others": 5
}

# Mapping of internal variable names to display names
display_labels = {
    'init_albumin': 'Albumin (g/dL)',
    'init_inr': 'INR',
    'init_bilirubin': 'Bilirubin (mg/dL)',
    'init_serum_creat': 'Creatinine (mg/dL)',
    'logafp': 'AFP (ng/mL)',
    'maxtumorsize': 'Largest Tumor Size (cm)',
    'sumtumoursize': 'Total Tumor Diameter (cm)',
    'HEwaitlist': 'Hepatic Encephalopathy',
    't2diabetes': 'Type 2 Diabetes Mellitus',
    'maxtumornumber': 'Tumor Number',
    'newdefinition': 'Underlying Liver Disease',
    'age60': 'Age ≥ 60',
    'obese': 'Obesity',
    'listascites': 'Ascites',
    #'ethnicitycoded': 'Ethnicity'
}

# Streamlit UI
st.title("HCC Survival Calculator")

# 1. Get user input for all 14 variables (customize to your actual features)
user_input = {}
for col in feature_names:
    label = display_labels.get(col, col)
    if col in binary_vars:
        # Checkbox returns bool, convert to int (0/1)
        user_input[col] = int(st.checkbox(label, value=False))
    #elif col == 'ethnicitycoded':
        #choice = st.selectbox(label, options=list(ethnicity_options.keys()))
        #user_input[col] = ethnicity_options[choice]
    elif col == 'newdefinition':
        choice = st.selectbox(label, options=list(newdefinition_options.keys()))
        user_input[col] = newdefinition_options[choice]
    elif col == 'logafp':
        continue  # skip logAFP, we will compute it from raw AFP input
    else:
        user_input[col] = st.number_input(label, value=0.00)

# Now add a raw AFP input (for the user)
raw_afp = st.number_input(display_labels['logafp'], min_value=0.0, value=0.0)

# Compute logAFP internally and add to inputs
user_input['logafp'] = np.log(raw_afp)

for k in user_input.keys():
    if isinstance(k, list):
        st.error(f"Key {k} is a list! This will cause errors.")

for k,v in user_input.items():
    if k == "maxtumornumber":
        if v.is_integer() == True:
            continue
        else:
            st.error(f"Tumour number should be an integer")


if st.button("Calculate Survival"):
    x_input = pd.DataFrame([user_input])
    x_transformed = x_mapper.transform(x_input).astype('float32')
    # Resultant dataset
    surv = model.predict_surv_df(x_transformed)


 # Extract survival probabilities at specific times
    def get_survival_prob(month):
    # Find the nearest row index to the desired month
        pos = np.argmin(np.abs(surv.index - month))
        actual_time = surv.index[pos]       # nearest available time
        prob = surv.iloc[pos, 0]            # survival prob at that row
        return actual_time, prob

    times_of_interest = [12, 36, 60]
    st.subheader("Predicted survival probabilities:")
    for t in times_of_interest:
        used_time, prob = get_survival_prob(t)
        st.write(f"**{int(round(used_time/12))}-year survival probability** "
                f"`{prob:.2%}`")

# Proper stepwise survival curve plot
    st.subheader("Predicted Survival Curve")
    fig, ax = plt.subplots()
    ax.step(surv.index, surv.iloc[:, 0], where='post', label='Predicted Survival')
    ax.set_xlabel('Time (months)')
    ax.set_ylabel('Survival Probability')
    ax.set_ylim(0, 1.05)
    ax.set_xlim(0, 120)
    ax.set_title('Predicted Survival Curve')
    ax.grid(False)
    ax.legend()
    max_time = int(np.ceil(surv.index.max() / 12) * 12)
    ax.set_xticks(range(0, max_time, 12))
    st.pyplot(fig)
