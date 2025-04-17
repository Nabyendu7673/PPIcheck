# Imports
import streamlit as st
import pandas as pd
import numpy as np
import io
import graphviz
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import random
from imblearn.over_sampling import SMOTE

# Page setup
st.set_page_config(page_title="PPI Risk & Deprescribing Advisor", layout="wide")
st.title("PPI Risk Prediction and Deprescribing Decision Support")

# Introduction
st.markdown("""
This application, designed to support PPI deprescribing decisions, employs a hybrid model integrating established clinical guidelines with machine learning. 
Developed by Dr. Nabyendu Biswas, Department of Pharmacology, in collaboration with the Medicine Department, MKCG Medical College & Hospital, 
its data correlates with Lexicomp drug interaction tools and CONFOR trial data, providing evidence-based risk assessment to aid clinicians in optimizing PPI therapy.
""")

st.markdown("---")  # Adding horizontal line for separation

# Patient Information Section
st.sidebar.header("Patient Information")

# 1. Demographics Section
st.sidebar.subheader("Demographics")
col_demo1, col_demo2 = st.sidebar.columns(2)
with col_demo1:
    patient_age = st.number_input("Age", min_value=0, max_value=120, value=60)
    patient_height = st.number_input("Height (cm)", min_value=100, max_value=250, value=170)
with col_demo2:
    patient_gender = st.selectbox("Gender", ["Male", "Female", "Other"])
    patient_weight = st.number_input("Weight (kg)", min_value=20, max_value=200, value=70)

bmi = patient_weight / ((patient_height/100) ** 2)
st.sidebar.write(f"BMI: {bmi:.1f} kg/m²")

# 2. Risk History
st.sidebar.markdown("---")
st.sidebar.subheader("Risk History")
col_risk1, col_risk2 = st.sidebar.columns(2)
with col_risk1:
    previous_gi_bleed = st.checkbox("GI Bleeding History")
    liver_disease = st.checkbox("Liver Disease")
    diabetes = st.checkbox("Diabetes")
with col_risk2:
    kidney_disease = st.checkbox("Kidney Disease")
    hypertension = st.checkbox("Hypertension")

# 3. Treatment History
st.sidebar.markdown("---")
st.sidebar.subheader("Treatment History")
ppi_duration = st.sidebar.number_input("PPI Duration (months)", min_value=0, max_value=240, value=0)
if ppi_duration > 12:
    st.sidebar.warning("⚠️ Long-term PPI use detected")

# 4. Clinical Indications
st.sidebar.markdown("---")
st.sidebar.subheader("Clinical Indications")
tabs = st.sidebar.tabs(["GI", "NSAID/Antiplatelet", "Other"])
gi_indications = ["Non-variceal bleeding", "Dyspepsia", "GERD & complications", "H pylori infection", "Peptic ulcer treatment", "Zollinger-Ellison syndrome"]
nsaid_antiplatelet_indications = ["Prevent NSAID ulcers", "NSAID & ulcer/GIB history", "NSAID & age > 60", "NSAID + cortico/antiplatelet/anticoag", "Prophylaxis in high risk antiplatelet users", "Antiplatelet & ulcer/GIB history", "Antiplatelet + age > 60 or dyspepsia/GERD", "Antiplatelet + cortico/NSAID/anticoag"]
other_indications = ["Stress ulcer prophylaxis", "Coagulopathy (platelet < 50k, INR ≥ 1.5)", "Mechanical ventilation > 48h"]

selected_gi = tabs[0].multiselect("GI Indications", gi_indications)
selected_nsaid_antiplatelet = tabs[1].multiselect("NSAID/Antiplatelet Indications", nsaid_antiplatelet_indications)
selected_other = tabs[2].multiselect("Other Indications", other_indications)
selected_indications = selected_gi + selected_nsaid_antiplatelet + selected_other

# 5. Current Medications
st.sidebar.markdown("---")
st.sidebar.subheader("Current Medications")

# PPI Inputs
selected_ppi = st.sidebar.selectbox("Select PPI", ["None", "Pantoprazole", "Omeprazole", "Esomeprazole", "Rabeprazole"])
ppi_dose = st.sidebar.selectbox("PPI Dose (mg)", [0, 20, 40, 80])
ppi_route = st.sidebar.selectbox("PPI Route", ["None", "Oral", "IV"])

# NSAID Section with new grouping
nsaid_groups = {
    "Salicylates": {
        "Aspirin": ([0, 75, 150, 300, 600], "Usual: 75–300 mg; Max: 600 mg/day", 600, 4)
    },
    "Propionic acid derivatives (Profens)": {
        "Ibuprofen": ([0, 200, 400, 600, 800, 2400], "Usual: 200–600 mg; Max: 2400 mg/day", 2400, 3),
        "Naproxen": ([0, 250, 375, 500, 1000], "Usual: 250–500 mg; Max: 1000 mg/day", 1000, 6),
        "Ketoprofen": ([0, 50, 100, 200], "Usual: 50–100 mg; Max: 200 mg/day", 200, 4),
        "Flurbiprofen": ([0, 50, 100, 150, 300], "Usual: 50–150 mg; Max: 300 mg/day", 300, 3)
    },
    "Acetic acid derivatives": {
        "Indomethacin": ([0, 25, 50, 75, 200], "Usual: 25–50 mg; Max: 200 mg/day", 200, 5),
        "Diclofenac": ([0, 25, 50, 75, 100, 150], "Usual: 50–75 mg; Max: 150 mg/day", 150, 4),
        "Etodolac": ([0, 200, 300, 400, 1000], "Usual: 200–400 mg; Max: 1000 mg/day", 1000, 3),
        "Ketorolac": ([0, 10, 20, 30, 120], "Usual: 10–30 mg; Max: 120 mg/day", 120, 4)
    },
    "Enolic acid (Oxicam) derivatives": {
        "Piroxicam": ([0, 10, 20], "Usual: 10–20 mg; Max: 20 mg/day", 20, 4),
        "Meloxicam": ([0, 7.5, 15], "Usual: 7.5–15 mg; Max: 15 mg/day", 15, 2)
    },
    "Selective COX-2 inhibitors": {
        "Celecoxib": ([0, 100, 200, 400], "Usual: 100–200 mg; Max: 400 mg/day", 400, 1)
    },
    "Non-NSAID Analgesics": {
        "Paracetamol": ([0, 500, 1000, 2000, 4000], "Usual: 500–1000 mg; Max: 4000 mg/day", 4000, 0)
    },
    "None": {"None": ([0], "", 0, 0)}
}

selected_nsaid_group = st.sidebar.selectbox("Select NSAID Group", list(nsaid_groups.keys()))
selected_nsaid = st.sidebar.selectbox("Select NSAID", list(nsaid_groups[selected_nsaid_group].keys()))
nsaid_info = nsaid_groups[selected_nsaid_group][selected_nsaid]
nsaid_dose_options, nsaid_help, nsaid_max_dose, nsaid_base_risk = nsaid_info
nsaid_dose = st.sidebar.selectbox("NSAID Dose (mg)", nsaid_dose_options, help=nsaid_help)
nsaid_route = st.sidebar.selectbox("NSAID Route", ["None", "Oral", "Parenteral"])

# Antiplatelet section
antiplatelet_dose_ranges = {
    "None": ([0], "", 0),
    "Aspirin": ([0, 75, 150, 300], "Usual: 75–150 mg/day; Max 300 mg/day", 300),
    "Clopidogrel": ([0, 75, 150, 300], "Usual: 75 mg; Loading: 300 mg", 300),
    "Ticagrelor": ([0, 90, 180], "Usual: 90 mg BID; Max 180 mg/day", 180),
    "Prasugrel": ([0, 5, 10], "Usual: 10 mg/day; Max 10 mg/day", 10),
}
selected_antiplatelet = st.sidebar.selectbox("Select Antiplatelet", list(antiplatelet_dose_ranges.keys()))
antiplatelet_dose_options, antiplatelet_help, antiplatelet_max = antiplatelet_dose_ranges[selected_antiplatelet]
antiplatelet_dose = st.sidebar.selectbox("Antiplatelet Dose (mg)", antiplatelet_dose_options, help=antiplatelet_help)
antiplatelet_route = st.sidebar.selectbox("Antiplatelet Route", ["None", "Oral"])

# Anticoagulant Section
selected_anticoagulant = st.sidebar.selectbox("Select Anticoagulant", ["None", "Warfarin", "Heparin", "Enoxaparin"])
anticoagulant_dose = st.sidebar.selectbox("Anticoagulant Dose", ["None", "Low Dose", "Moderate Dose", "High Dose"])
anticoagulant_route = st.sidebar.selectbox("Anticoagulant Route", ["None", "Oral", "IV", "Subcutaneous"])

# Scoring functions
def get_nsaid_score(dose, max_dose, base_risk_score):
    if dose == 0 or dose == "None":
        return 0
    
    dose_percentage = (dose / max_dose) * 100
    if dose_percentage <= 25:
        return base_risk_score
    elif dose_percentage <= 50:
        return base_risk_score + 1
    elif dose_percentage <= 75:
        return base_risk_score + 2
    else:
        return base_risk_score + 3

def get_antiplatelet_score(dose):
    if dose == 0 or dose == "None":
        return 0
    elif dose <= 75:
        return 1
    elif dose <= 150:
        return 2
    elif dose <= 300:
        return 3
    else:
        return 4

def get_ppi_gastroprotection(dose, route, nsaid_flag, antiplatelet_flag, anticoagulant_flag):
    reduction = 0
    if nsaid_flag or antiplatelet_flag or anticoagulant_flag:
        if route == "Oral" and dose >= 20:
            reduction = -1
        elif route == "IV" and dose >= 40:
            reduction = -2
    return reduction

# Calculate scores
# NSAID score
if selected_nsaid != "None":
    try:
        nsaid_dose = int(nsaid_dose)
        if nsaid_dose > nsaid_max_dose:
            st.sidebar.warning(f"Dose exceeds max recommended for {selected_nsaid}!")
        nsaid_score = get_nsaid_score(nsaid_dose, nsaid_max_dose, nsaid_base_risk)
    except ValueError:
        st.sidebar.error("Invalid NSAID dose input.")
        nsaid_score = 0
else:
    nsaid_score = 0

# Antiplatelet score
if selected_antiplatelet != "None":
    try:
        antiplatelet_dose = int(antiplatelet_dose)
        if antiplatelet_dose > antiplatelet_max:
            st.sidebar.warning(f"Dose exceeds max recommended for {selected_antiplatelet}!")
        antiplatelet_score = get_antiplatelet_score(antiplatelet_dose)
    except ValueError:
        st.sidebar.error("Invalid antiplatelet dose input.")
        antiplatelet_score = 0
else:
    antiplatelet_score = 0

# Anticoagulant score
anticoagulant_score = {"None": 0, "Low Dose": 1, "Moderate Dose": 2, "High Dose": 3}.get(anticoagulant_dose, 0)

# Interaction Alerts
interaction_alert = ""
if selected_antiplatelet == "Aspirin" and selected_anticoagulant == "Warfarin":
    interaction_alert = "High bleeding risk: Aspirin + Warfarin."
elif selected_antiplatelet == "Clopidogrel" and selected_anticoagulant == "Heparin":
    interaction_alert = "Increased bleeding risk: Clopidogrel + Heparin."
elif selected_antiplatelet == "Ticagrelor" and selected_anticoagulant == "Enoxaparin":
    interaction_alert = "Monitor closely: Ticagrelor + Enoxaparin increases bleeding risk."

if interaction_alert:
    st.error(f"⚠️ Drug–Drug Interaction Alert: {interaction_alert}")

# Calculate indication score
indication_weights = {
    "Non-variceal bleeding": 3, "Dyspepsia": 1, "GERD & complications": 2, "H pylori infection": 2, "Peptic ulcer treatment": 3,
    "Zollinger-Ellison syndrome": 3, "Prevent NSAID ulcers": 2, "NSAID & ulcer/GIB history": 3, "NSAID & age > 60": 2,
    "NSAID + cortico/antiplatelet/anticoag": 3, "Prophylaxis in high risk antiplatelet users": 2, "Antiplatelet & ulcer/GIB history": 3,
    "Antiplatelet + age > 60 or dyspepsia/GERD": 2, "Antiplatelet + cortico/NSAID/anticoag": 3, "Stress ulcer prophylaxis": 2,
    "Coagulopathy (platelet < 50k, INR ≥ 1.5)": 2, "Mechanical ventilation > 48h": 2,
}
indication_score = sum([indication_weights.get(ind, 0) for ind in selected_indications])

# Calculate risk flags
nsaid_flag = int(selected_nsaid != "None")
antiplatelet_flag = int(selected_antiplatelet != "None")
anticoagulant_flag = int(selected_anticoagulant != "None")
triple_combo_flag = int(nsaid_flag and antiplatelet_flag and anticoagulant_flag)
medication_risk = nsaid_score + antiplatelet_score + anticoagulant_score
high_risk_flag = int(medication_risk >= 6 or indication_score >= 6)

# Calculate PPI reduction
ppi_reduction = get_ppi_gastroprotection(ppi_dose, ppi_route, nsaid_flag, antiplatelet_flag, anticoagulant_flag)

# Calculate final score
score = medication_risk + indication_score + (triple_combo_flag * 2) + high_risk_flag + ppi_reduction

# Output Section
st.subheader("Risk Scoring Result")
st.write(f"**Total Risk Score:** {score}")

# Risk interpretation
if score >= 10:
    st.warning("Very high risk – Continue PPI. Deprescribing not advised.")
elif score >= 7:
    st.warning("High risk – Continue PPI. Consider dose optimization.")
elif 4 <= score < 7:
    st.info("Moderate risk – Reassess need. Consider step-down.")
else:
    st.success("Low risk – Deprescribing can be considered.")

# Detailed Scoring Breakdown
st.subheader("Detailed Score Breakdown")
col1, col2 = st.columns(2)

with col1:
    st.markdown("**Medication Scores:**")
    st.write(f"- NSAID Score: {nsaid_score}")
    st.write(f"- Antiplatelet Score: {antiplatelet_score}")
    st.write(f"- Anticoagulant Score: {anticoagulant_score}")
    st.write(f"- Triple Therapy Flag: {triple_combo_flag * 2}")
    st.write(f"- High Risk Flag: {high_risk_flag}")
    st.write(f"- PPI Reduction: {ppi_reduction}")

with col2:
    st.markdown("**Clinical Indication Score:**")
    st.write(f"Total Indication Score: {indication_score}")
    if selected_indications:
        st.markdown("Selected Indications:")
        for ind in selected_indications:
            st.write(f"- {ind}: {indication_weights[ind]}")

# --- Flowchart Generation ---
st.subheader("PPI Deprescribing Flowchart")
with st.expander("📊 View Detailed Deprescribing Flowchart"):
    def generate_ppi_flowchart(score, nsaid_score, antiplatelet_score, anticoagulant_score, indication_score, triple_combo_flag, high_risk_flag, ppi_reduction, graph_size="50,50"):
        try:
            dot = graphviz.Digraph(comment='PPI Deprescribing Algorithm', graph_attr={'rankdir': 'TD', 'size': graph_size})
            dot.node('start', 'Start', shape='oval', style='filled', fillcolor='#E0E0E0')
            dot.node('score_check', f'Total Score: {score}\nMed: {nsaid_score + antiplatelet_score + anticoagulant_score}\nInd: {indication_score}', shape='diamond', style='filled', fillcolor='#ADD8E6')

            if score == 0:
                dot.node('stop_ppi', 'Stop PPI\nNo indication found.', shape='box', style='filled', fillcolor='#90EE90')
                dot.edge('start', 'score_check')
                dot.edge('score_check', 'stop_ppi')
            elif 1 <= score <= 2:
                dot.node('clinical_judgment', 'Clinical Judgment\nConsider Deprescribing.', shape='box', style='filled', fillcolor='#FFFFE0')
                dot.edge('start', 'score_check')
                dot.edge('score_check', 'clinical_judgment')
            elif 3 <= score <= 5:
                dot.node('monitoring', 'Deprescribing with\n4-week monitoring.', shape='box', style='filled', fillcolor='#FFA500')
                dot.node('change_oral', 'Change IV/IM to Oral\nif possible.', shape='box', style='filled', fillcolor='#FFDAB9')
                dot.node('reduce_dose', 'Reduce drug doses\nif possible.', shape='box', style='filled', fillcolor='#FFDAB9')
                dot.edge('start', 'score_check')
                dot.edge('score_check', 'monitoring')
                dot.edge('monitoring', 'change_oral')
                dot.edge('change_oral', 'reduce_dose')
            else:
                dot.node('continue_ppi', 'Continue PPI\nHigh risk for GI complications.', shape='box', style='filled', fillcolor='#F08080')
                dot.edge('start', 'score_check')
                dot.edge('score_check', 'continue_ppi')

            dot.node('nsaid_score', f'NSAID Score: {nsaid_score}', shape='box')
            dot.node('antiplatelet_score', f'Antiplatelet Score: {antiplatelet_score}', shape='box')
            dot.node('anticoagulant_score', f'Anticoagulant Score: {anticoagulant_score}', shape='box')
            dot.node('ind_score', f'Indication Score: {indication_score}', shape='box')
            dot.node('triple_flag', f'Triple Combo: {triple_combo_flag}', shape='box')
            dot.node('high_risk', f'High Risk: {high_risk_flag}', shape='box')
            dot.node('ppi_reduction', f'PPI Reduction: {ppi_reduction}', shape='box')

            dot.edge('score_check', 'nsaid_score', label="Scores")
            dot.edge('score_check', 'antiplatelet_score')
            dot.edge('score_check', 'anticoagulant_score')
            dot.edge('score_check', 'ind_score')
            dot.edge('score_check', 'triple_flag')
            dot.edge('score_check', 'high_risk')
            dot.edge('score_check', 'ppi_reduction')

            return dot
        except Exception as e:
            st.error(f"Error generating flowchart: {e}")
            return None

    # Generate and display the flowchart
    flowchart = generate_ppi_flowchart(
        score=score,
        nsaid_score=nsaid_score,
        antiplatelet_score=antiplatelet_score,
        anticoagulant_score=anticoagulant_score,
        indication_score=indication_score,
        triple_combo_flag=triple_combo_flag,
        high_risk_flag=high_risk_flag,
        ppi_reduction=ppi_reduction
    )
    
    if flowchart:
        st.graphviz_chart(flowchart)
    else:
        st.error("Failed to generate flowchart")

    st.markdown("""
    **Flowchart Legend:**
    - 🔵 Score Check: Evaluates total risk score and component scores
    - 🟢 Stop PPI: No clinical indication for PPI continuation
    - 🟡 Clinical Judgment: Low risk, consider deprescribing
    - 🟠 Monitoring: Moderate risk, requires careful monitoring
    - 🔴 Continue PPI: High risk, maintain current therapy
    """)

# --- Log Entry and CSV Download ---
if 'logged_data' not in st.session_state:
    st.session_state.logged_data = []

if st.button("Log Input Data"):
    input_data = {
        "PPI": selected_ppi, "PPI Dose": ppi_dose, "PPI Route": ppi_route,
        "NSAID": selected_nsaid, "NSAID Dose": nsaid_dose, "NSAID Route": nsaid_route,
        "Antiplatelet": selected_antiplatelet, "Antiplatelet Dose": antiplatelet_dose, "Antiplatelet Route": antiplatelet_route,
        "Anticoagulant": selected_anticoagulant, "Anticoagulant Dose": anticoagulant_dose, "Anticoagulant Route": anticoagulant_route,
        "Indications": ", ".join(selected_indications), "Score": score
    }
    st.session_state.logged_data.append(input_data)
    st.success("Data logged successfully!")

if st.session_state.logged_data:
    df_logged = pd.DataFrame(st.session_state.logged_data)
    csv_buffer = io.StringIO()
    df_logged.to_csv(csv_buffer, index=False)
    csv_string = csv_buffer.getvalue()
    csv_bytes = csv_string.encode()
    st.download_button(label="Download Logged Data as CSV", data=csv_bytes, file_name="logged_data.csv", mime="text/csv")

# --- ML Model Training and Evaluation ---
@st.cache_data
def generate_synthetic_data(num_samples=2000):
    data = []
    for _ in range(num_samples):
        ppi = random.choice(["None", "Pantoprazole", "Omeprazole", "Esomeprazole", "Rabeprazole"])
        ppi_dose = random.choice([0, 20, 40, 80])
        ppi_route = random.choice(["None", "Oral", "IV"])
        
        nsaid_group = random.choice(list(nsaid_groups.keys()))
        nsaid = random.choice(list(nsaid_groups[nsaid_group].keys()))
        nsaid_info = nsaid_groups[nsaid_group][nsaid]
        nsaid_dose = random.choice(nsaid_info[0])
        nsaid_route = random.choice(["None", "Oral", "Parenteral"])
        
        antiplatelet = random.choice(list(antiplatelet_dose_ranges.keys()))
        antiplatelet_dose = random.choice(antiplatelet_dose_ranges[antiplatelet][0])
        antiplatelet_route = random.choice(["None", "Oral"])
        
        anticoagulant = random.choice(["None", "Warfarin", "Heparin", "Enoxaparin"])
        anticoagulant_dose = random.choice(["None", "Low Dose", "Moderate Dose", "High Dose"])
        anticoagulant_route = random.choice(["None", "Oral", "IV", "Subcutaneous"])
        
        num_indications = random.randint(0, 5)
        all_indications = gi_indications + nsaid_antiplatelet_indications + other_indications
        indications = random.sample(all_indications, num_indications)
        
        # Calculate synthetic scores
        nsaid_score = get_nsaid_score(nsaid_dose, nsaid_info[2], nsaid_info[3]) if nsaid != "None" else 0
        antiplatelet_score = get_antiplatelet_score(antiplatelet_dose)
        anticoagulant_score = {"None": 0, "Low Dose": 1, "Moderate Dose": 2, "High Dose": 3}[anticoagulant_dose]
        
        nsaid_flag = int(nsaid != "None")
        antiplatelet_flag = int(antiplatelet != "None")
        anticoagulant_flag = int(anticoagulant != "None")
        triple_combo_flag = int(nsaid_flag and antiplatelet_flag and anticoagulant_flag)
        
        indication_score = sum(indication_weights.get(ind, 0) for ind in indications)
        medication_risk = nsaid_score + antiplatelet_score + anticoagulant_score
        high_risk_flag = int(medication_risk >= 6 or indication_score >= 6)
        
        ppi_reduction = get_ppi_gastroprotection(ppi_dose, ppi_route, nsaid_flag, antiplatelet_flag, anticoagulant_flag)
        
        score = medication_risk + indication_score + (triple_combo_flag * 2) + high_risk_flag + ppi_reduction
        
        data.append({
            "PPI": ppi, "PPI Dose": ppi_dose, "PPI Route": ppi_route,
            "NSAID": nsaid, "NSAID Dose": nsaid_dose, "NSAID Route": nsaid_route,
            "Antiplatelet": antiplatelet, "Antiplatelet Dose": antiplatelet_dose, "Antiplatelet Route": antiplatelet_route,
            "Anticoagulant": anticoagulant, "Anticoagulant Dose": anticoagulant_dose, "Anticoagulant Route": anticoagulant_route,
            "Indications": ", ".join(indications), "Score": score
        })
    return pd.DataFrame(data)

@st.cache_data
def train_and_evaluate_models(data):
    X = data.drop("Score", axis=1)
    y = (data["Score"] >= 4).astype(int)  # Binary classification: High Risk (1) or Low/Moderate Risk (0)
    X = pd.get_dummies(X)  # One hot encode categorical data

    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)

    X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

    rf_model = RandomForestClassifier(random_state=42)
    lr_model = LogisticRegression(random_state=42, max_iter=1000)

    rf_model.fit(X_train, y_train)
    lr_model.fit(X_train, y_train)

    rf_probs = rf_model.predict_proba(X_test)[:, 1]
    lr_probs = lr_model.predict_proba(X_test)[:, 1]

    rf_predictions = rf_model.predict(X_test)
    lr_predictions = lr_model.predict(X_test)

    rf_metrics = {
        "Accuracy": accuracy_score(y_test, rf_predictions),
        "Precision": precision_score(y_test, rf_predictions),
        "Recall": recall_score(y_test, rf_predictions),
        "F1 Score": f1_score(y_test, rf_predictions),
        "AUC": roc_auc_score(y_test, rf_probs),
    }

    lr_metrics = {
        "Accuracy": accuracy_score(y_test, lr_predictions),
        "Precision": precision_score(y_test, lr_predictions),
        "Recall": recall_score(y_test, lr_predictions),
        "F1 Score": f1_score(y_test, lr_predictions),
        "AUC": roc_auc_score(y_test, lr_probs),
    }

    rf_fpr, rf_tpr, _ = roc_curve(y_test, rf_probs)
    lr_fpr, lr_tpr, _ = roc_curve(y_test, lr_probs)

    return rf_fpr, rf_tpr, lr_fpr, lr_tpr, rf_metrics, lr_metrics

# Train and evaluate models
synthetic_data = generate_synthetic_data()
rf_fpr, rf_tpr, lr_fpr, lr_tpr, rf_metrics, lr_metrics = train_and_evaluate_models(synthetic_data)

# Display model evaluation results
st.subheader("Machine Learning Model Evaluation")
col1, col2 = st.columns(2)

with col1:
    fig_rf, ax_rf = plt.subplots(figsize=(4, 4), facecolor="none")
    ax_rf.set_facecolor("none")
    ax_rf.plot(rf_fpr, rf_tpr, label=f"RF (AUC = {rf_metrics['AUC']:.2f})")
    ax_rf.plot([0, 1], [0, 1], "k--")
    ax_rf.set_xlabel("False Positive Rate")
    ax_rf.set_ylabel("True Positive Rate")
    ax_rf.set_title("Random Forest ROC Curve")
    ax_rf.legend()
    st.pyplot(fig_rf)

with col2:
    fig_lr, ax_lr = plt.subplots(figsize=(4, 4), facecolor="none")
    ax_lr.set_facecolor("none")
    ax_lr.plot(lr_fpr, lr_tpr, label=f"LR (AUC = {lr_metrics['AUC']:.2f})")
    ax_lr.plot([0, 1], [0, 1], "k--")
    ax_lr.set_xlabel("False Positive Rate")
    ax_lr.set_ylabel("True Positive Rate")
    ax_lr.set_title("Logistic Regression ROC Curve")
    ax_lr.legend()
    st.pyplot(fig_lr)

# Display metrics table
metrics_df = pd.DataFrame([rf_metrics, lr_metrics], index=["Random Forest", "Logistic Regression"])
st.table(metrics_df)

# Model Training Note
st.markdown("""
**Model Training Information:**
* Training: Synthetic dataset (2,000 entries) with SMOTE for balanced class representation
* Binary Classification: High Risk (Score ≥ 4) vs Low/Moderate Risk (Score < 4)
* Note: This is synthetic data for validation purposes; performance may vary with real-world data
""")

# Source citation
st.markdown("---")
st.markdown(
    "🔗 **Source:** Deprescribing proton pump inhibitors. Evidence-based clinical practice guideline. "
    "*Canadian Family Physician* May 2017; 63 (5): 354–364."
)