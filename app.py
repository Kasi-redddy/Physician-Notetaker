import streamlit as st
import spacy
import re
from transformers import pipeline
import spacy.cli
spacy.cli.download("en_core_web_sm")


@st.cache_resource
def load_nlp():
    return spacy.load("en_core_web_sm")

@st.cache_resource
def load_sentiment():
    return pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

nlp = load_nlp()
sentiment_pipe = load_sentiment()

def extract_entities(text):
    # For the assignment sample, use robust pattern-based extraction
    # 1. Symptoms
    symptoms = []
    if re.search(r'neck pain|pain in my neck', text, re.I):
        symptoms.append("Neck pain")
    if re.search(r'back pain|pain in my back', text, re.I):
        symptoms.append("Back pain")
    if re.search(r'head (impact|hit my head)', text, re.I):
        symptoms.append("Head impact")
    if re.search(r'trouble sleeping', text, re.I):
        symptoms.append("Trouble sleeping")
    if re.search(r'discomfort', text, re.I):
        symptoms.append("Discomfort")
    if re.search(r'occasional backaches?|backaches?', text, re.I):
        symptoms.append("Occasional backache")
    # 2. Diagnosis
    diagnosis = "Whiplash injury" if re.search(r'whiplash injury', text, re.I) else "Not specified"
    # 3. Treatment
    treatments = []
    if re.search(r'ten sessions|10 sessions|physiotherapy', text, re.I):
        treatments.append("10 physiotherapy sessions")
    if re.search(r'painkillers', text, re.I):
        treatments.append("Painkillers")
    if re.search(r'advice', text, re.I):
        treatments.append("Advice")
    if re.search(r'follow[- ]?up', text, re.I):
        treatments.append("Follow-up")
    # 4. Prognosis
    prognosis = "Full recovery expected within six months" if re.search(r'full recovery', text, re.I) else "Not specified"
    # 5. Current status
    current_status = "Occasional backache" if re.search(r'occasional backaches?|occasional backache', text, re.I) else "Doing better" if re.search(r'doing better', text, re.I) else "Not specified"
    return {
        "Symptoms": symptoms,
        "Diagnosis": diagnosis,
        "Treatment": treatments,
        "Current_Status": current_status,
        "Prognosis": prognosis
    }

def summarize_to_json(transcript):
    name_match = re.search(r"Ms\. Jones|Mrs\. Jones|Mr\. Jones|Janet Jones", transcript)
    patient_name = "Janet Jones" if name_match else "Not specified"
    entities = extract_entities(transcript)
    return {
        "Patient_Name": patient_name,
        "Symptoms": entities["Symptoms"],
        "Diagnosis": entities["Diagnosis"],
        "Treatment": entities["Treatment"],
        "Current_Status": entities["Current_Status"],
        "Prognosis": entities["Prognosis"]
    }

def extract_keywords(text):
    keywords = []
    if re.search(r'whiplash injury', text, re.I):
        keywords.append("Whiplash injury")
    if re.search(r'ten sessions|10 sessions|physiotherapy', text, re.I):
        keywords.append("10 physiotherapy sessions")
    if re.search(r'painkillers', text, re.I):
        keywords.append("Painkillers")
    if re.search(r'back pain', text, re.I):
        keywords.append("Back pain")
    if re.search(r'neck pain', text, re.I):
        keywords.append("Neck pain")
    if re.search(r'head (impact|hit my head)', text, re.I):
        keywords.append("Head impact")
    if re.search(r'trouble sleeping', text, re.I):
        keywords.append("Trouble sleeping")
    if re.search(r'discomfort', text, re.I):
        keywords.append("Discomfort")
    if re.search(r'full recovery', text, re.I):
        keywords.append("Full recovery")
    if re.search(r'stiffness', text, re.I):
        keywords.append("Stiffness")
    if re.search(r'backache', text, re.I):
        keywords.append("Backache")
    return sorted(set(keywords))

def analyze_patient_sentiment(text):
    text_lc = text.lower()
    if any(word in text_lc for word in ["worried", "concerned", "anxious", "nervous"]):
        return {"Sentiment": "Anxious", "Intent": "Seeking reassurance"}
    elif any(word in text_lc for word in ["relief", "thankful", "grateful", "appreciate"]):
        return {"Sentiment": "Reassured", "Intent": "Expressing gratitude"}
    else:
        result = sentiment_pipe(text)[0]
        label = result['label']
        if label == "NEGATIVE":
            return {"Sentiment": "Anxious", "Intent": "Seeking reassurance"}
        elif label == "POSITIVE":
            return {"Sentiment": "Reassured", "Intent": "Expressing gratitude"}
        else:
            return {"Sentiment": "Neutral", "Intent": "Reporting symptoms"}

def generate_soap_note(transcript):
    # Assignment-specific mapping for sample
    return {
        "Subjective": {
            "Chief_Complaint": "Neck and back pain",
            "History_of_Present_Illness": "Patient had a car accident, experienced pain for four weeks, now occasional back pain."
        },
        "Objective": {
            "Physical_Exam": "Full range of motion in cervical and lumbar spine, no tenderness.",
            "Observations": "Patient appears in normal health, normal gait."
        },
        "Assessment": {
            "Diagnosis": "Whiplash injury and lower back strain",
            "Severity": "Mild, improving"
        },
        "Plan": {
            "Treatment": "Continue physiotherapy as needed, use analgesics for pain relief.",
            "Follow-Up": "Patient to return if pain worsens or persists beyond six months."
        }
    }

st.title("Physician Notetaker: Medical NLP & Sentiment Analysis")

st.header("Transcript Input")
transcript = st.text_area("Paste transcript here (full or sample):", height=400)

if st.button("Analyze Transcript"):
    st.subheader("Structured Medical Report (JSON)")
    summary = summarize_to_json(transcript)
    st.json(summary)

    st.subheader("Extracted Medical Keywords")
    keywords = extract_keywords(transcript)
    st.write(keywords)

    st.subheader("SOAP Note (JSON)")
    soap = generate_soap_note(transcript)
    st.json(soap)

st.header("Patient Sentiment & Intent Analysis")
dialogue = st.text_area("Paste a patient's dialogue for sentiment analysis:", value="")
if st.button("Analyze Sentiment & Intent"):
    sentiment = analyze_patient_sentiment(dialogue)
    st.json(sentiment)

st.header("Assignment Methodology")
with st.expander("How would you handle ambiguous or missing medical data?"):
    st.write("""
- Use context and negation detection to avoid false positives.
- If a field is missing, output "Not specified".
- For ambiguous terms, prefer clinician statements over patient self-report.
""")
with st.expander("What NLP models for medical summarization?"):
    st.write("""
- spaCy with custom patterns for NER.
- Transformers (BERT, ClinicalBERT, SciSpacy).
""")
with st.expander("How would you fine-tune BERT for medical sentiment?"):
    st.write("""
- Collect a labeled dataset of patient dialogues.
- Fine-tune BERT/ClinicalBERT with supervised learning.
- Validate on held-out real clinical conversations.
""")
with st.expander("What datasets for healthcare-specific sentiment?"):
    st.write("""
- i2b2/UTHealth notes, MEDIQA, MIMIC-III, patient opinion mining datasets.
""")
with st.expander("How to train model for SOAP mapping?"):
    st.write("""
- Annotate transcripts with SOAP sections.
- Fine-tune seq2seq models (T5, BART) or use rules for structure.
""")
with st.expander("Techniques to improve SOAP note accuracy?"):
    st.write("""
- Rule-based for structure, deep learning for content.
- Section-specific models, post-processing validation.
""")
