import streamlit as st
import numpy as np
import joblib

# -----------------------------
# Load trained artifacts
# -----------------------------
model = joblib.load("final_ensemble.pkl")
features = joblib.load("feature_names.pkl")

NUM_FEATURES = len(features)
MAX_QUESTIONS = 20
CONFIDENCE_THRESHOLD = 0.95

# -----------------------------
# Page setup (Akinator style)
# -----------------------------
st.set_page_config(
    page_title="Disease Akinator",
    page_icon="🧠",
    layout="centered"
)

st.title("🧠 Disease Akinator")
st.caption("Think of your symptoms. I will try to guess your disease.")

# -----------------------------
# Session state initialization
# -----------------------------
if "user_state" not in st.session_state:
    st.session_state.user_state = np.zeros((1, NUM_FEATURES))
    st.session_state.asked = []
    st.session_state.done = False

# -----------------------------
# Helper functions
# -----------------------------
def select_next_question():
    entropies = {}
    for i, col in enumerate(features):
        if col in st.session_state.asked:
            continue
        p = 0.5  # neutral entropy assumption
        entropies[col] = -(p*np.log2(p+1e-9) + (1-p)*np.log2(1-p+1e-9))
    return max(entropies, key=entropies.get)

def update_user_state(idx, answer):
    mapping = {
        "Yes": 1.0,
        "Probably": 0.75,
        "Probably Not": 0.25,
        "No": 0.0
    }
    st.session_state.user_state[0, idx] = mapping[answer]

def check_prediction():
    probs = model.predict_proba(st.session_state.user_state)[0]
    best_idx = probs.argmax()
    return probs, probs[best_idx], model.classes_[best_idx]

# -----------------------------
# Progress bar
# -----------------------------
progress = min(len(st.session_state.asked) / MAX_QUESTIONS, 1.0)

# -----------------------------
# Main Akinator logic
# -----------------------------
if not st.session_state.done:

    question = select_next_question()
    q_index = features.index(question)

    st.subheader(f"❓ Do you have **{question}** ?")

    answer = st.radio(
        "Choose one:",
        ["Yes", "Probably", "Probably Not", "No"],
        horizontal=True
    )

    if st.button("Next ▶️"):
        update_user_state(q_index, answer)
        st.session_state.asked.append(question)

        probs, best_prob, best_disease = check_prediction()

        if best_prob >= CONFIDENCE_THRESHOLD or len(st.session_state.asked) >= MAX_QUESTIONS:
            st.session_state.done = True
            st.session_state.final_probs = probs
            st.session_state.final_disease = best_disease
            st.session_state.final_confidence = best_prob

# -----------------------------
# Final Answer Screen
# -----------------------------
else:
    st.success("🎯 Diagnosis Completed")

    st.markdown(f"""
    ## 🩺 Predicted Disease
    **{st.session_state.final_disease}**

    **Confidence:** {st.session_state.final_confidence*100:.2f}%
    """)

    top3_idx = st.session_state.final_probs.argsort()[-3:][::-1]

    st.markdown("### 🔍 Other Possible Matches")
    for idx in top3_idx:
        st.write(f"- {model.classes_[idx]} : {st.session_state.final_probs[idx]*100:.2f}%")

    st.warning(
        "⚠️ This tool is for educational purposes only. "
        "Please consult a medical professional for diagnosis."
    )

    if st.button("🔁 Start New Diagnosis"):
        st.session_state.user_state = np.zeros((1, NUM_FEATURES))
        st.session_state.asked = []
        st.session_state.done = False
