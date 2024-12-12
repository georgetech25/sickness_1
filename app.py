import streamlit as st
import joblib

st.set_page_config(
    page_icon='logo.jpeg',
    page_title='Self Diagnosis'
)

# Load the model and vectorizer
model = joblib.load('sickness_predictor_model.pkl')
vectorizer = joblib.load('symptoms_vectorizer.pkl')

# Extract all symptoms and sicknesses
all_symptoms = vectorizer.get_feature_names_out()
all_sicknesses = model.classes_

# Mappings for prescriptions, causes, and symptoms
prescription_mapping = {
    "Flu": "Rest, hydration, Paracetamol",
    "Migraine": "Painkillers, rest in a dark room",
    "Common Cold": "Antihistamines, warm water gargle",
    "Pneumonia": "Antibiotics, oxygen therapy",
    "Asthma": "Inhalers, avoid triggers, bronchodilators",
    "COVID-19": "Isolation, antiviral medications, rest, hydration",
    "Dengue Fever": "Paracetamol, fluid replacement",
    "Brain Tumor": "Surgical intervention, chemotherapy, radiation therapy",
    "Stroke": "Emergency medical care, blood thinners, physical therapy",
    "Allergic Reaction": "Antihistamines, epinephrine (for severe reactions), avoid allergens",
    "Tonsillitis": "Warm saltwater gargle, antibiotics (if bacterial), pain relievers",
    "Kidney Infection": "Antibiotics, hydration, pain relievers",
    "Lymphoma": "Chemotherapy, radiation therapy, targeted therapy",
    "Malaria": "Antimalarial drugs, hydration, fever management",
    "Tuberculosis": "Antitubercular drugs, long-term treatment regimen",
    "Food Poisoning": "Rehydration, rest, anti-nausea medications",
    "Multiple Sclerosis": "Immunomodulatory drugs, physical therapy, symptom management",
    "Gastritis": "Antacids, avoid spicy foods, proton pump inhibitors",
    "Meningitis": "Antibiotics (if bacterial), corticosteroids, hospitalization",
    "Hepatitis": "Rest, antiviral medications, avoid alcohol, liver support therapy",
}

causes_mapping = {
    "Flu": "Influenza virus, seasonal changes, weakened immunity",
    "Migraine": "Stress, lack of sleep, certain foods or smells",
    "Common Cold": "Rhinovirus, exposure to cold weather, low immunity",
    "Pneumonia": "Bacterial infection, viral infection, fungal infection",
    "Asthma": "Allergic reactions, environmental triggers, genetic factors",
    "COVID-19": "Coronavirus infection, close contact with infected persons",
    "Dengue Fever": "Mosquito bites (Aedes aegypti), tropical climates",
    "Brain Tumor": "Genetic mutations, radiation exposure",
    "Stroke": "Blocked blood flow to the brain, burst blood vessels",
    "Allergic Reaction": "Exposure to allergens (pollen, food, medication)",
    "Tonsillitis": "Bacterial or viral infection of the tonsils",
    "Kidney Infection": "Bacterial infection, untreated urinary tract infections",
    "Lymphoma": "Abnormal growth of lymphocytes, genetic predisposition",
    "Malaria": "Plasmodium parasite through mosquito bites",
    "Tuberculosis": "Mycobacterium tuberculosis infection, airborne transmission",
    "Food Poisoning": "Contaminated food or water, bacterial toxins",
    "Multiple Sclerosis": "Immune system attacks the central nervous system, genetic factors",
    "Gastritis": "Helicobacter pylori infection, overuse of NSAIDs, alcohol",
    "Meningitis": "Infection of the protective membranes of the brain, bacterial or viral",
    "Hepatitis": "Viral infections (Hepatitis A, B, C, etc.), alcohol abuse, certain medications, toxins",
}

symptoms_mapping = {
    "Flu": "Fever, cough, sore throat, muscle aches",
    "Migraine": "Severe headache, nausea, sensitivity to light",
    "Common Cold": "Runny nose, congestion, sneezing, mild fever",
    "Pneumonia": "Cough with phlegm, chest pain, shortness of breath",
    "Asthma": "Wheezing, shortness of breath, chest tightness",
    "COVID-19": "Fever, cough, loss of taste or smell, fatigue",
    "Dengue Fever": "High fever, severe joint pain, rash, bleeding gums",
    "Brain Tumor": "Persistent headache, vision problems, seizures",
    "Stroke": "Sudden numbness, confusion, difficulty speaking or walking",
    "Allergic Reaction": "Itching, swelling, difficulty breathing (severe cases)",
    "Tonsillitis": "Sore throat, swollen tonsils, difficulty swallowing",
    "Kidney Infection": "Back pain, fever, frequent urination with pain",
    "Lymphoma": "Swollen lymph nodes, fatigue, unexplained weight loss",
    "Malaria": "High fever, chills, sweating, headache",
    "Tuberculosis": "Persistent cough, night sweats, weight loss",
    "Food Poisoning": "Nausea, vomiting, diarrhea, abdominal cramps",
    "Multiple Sclerosis": "Muscle weakness, numbness, coordination issues",
    "Gastritis": "Stomach pain, nausea, bloating, loss of appetite",
    "Meningitis": "Severe headache, stiff neck, sensitivity to light",
    "Hepatitis": "Jaundice (yellowing of skin and eyes), fatigue, abdominal pain, nausea, dark urine",
}

# Streamlit app
st.title("Self Sickness Diagnosis App")
st.write("Enter your symptoms to get a diagnosis, prescription, possible causes, and related symptoms.")
st.image('testoscope.jpeg', width=200)

# Display all possible symptoms for user guidance
st.subheader("Possible Symptoms:")
st.write(", ".join(all_symptoms))

# User input
symptoms = st.text_area("Symptoms (e.g., fever, cough, fatigue):")

if st.button("Predict"):
    if symptoms.strip():
        # Vectorize input and predict
        symptoms_vec = vectorizer.transform([symptoms])
        sickness = model.predict(symptoms_vec)[0]
        prescription = prescription_mapping.get(sickness, "Consult a doctor.")
        causes = causes_mapping.get(sickness, "Unknown causes. Consult a doctor.")
        related_symptoms = symptoms_mapping.get(sickness, "No specific symptoms listed.")
        
        # Display results
        st.subheader(f"Predicted Sickness: {sickness}")
        st.write(f"Prescription: {prescription}")
        st.write(f"Possible Causes: {causes}")
        st.write(f"Related Symptoms: {related_symptoms}")
    else:
        st.error("Please enter symptoms.")
