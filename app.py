import streamlit as st
import numpy as np
import joblib

# =====================================================
# CONFIG
# =====================================================
st.set_page_config(
    page_title="Disease Akinator",
    page_icon="🧞",
    layout="centered"
)

MAX_QUESTIONS = 25
CONFIDENCE_THRESHOLD = 0.87

# =====================================================
# THEME STATE
# =====================================================
if "theme" not in st.session_state:
    st.session_state.theme = "light"

with st.sidebar:
    st.markdown("## 🎨 Theme")
    theme_choice = st.radio("Choose mode:", ["Light", "Dark"], index=0)
    st.session_state.theme = theme_choice.lower()

if st.session_state.theme == "dark":
    st.markdown("""
    <style>
    body, .stApp { background-color:#0e1117; color:#fafafa; }
    </style>
    """, unsafe_allow_html=True)
else:
    st.markdown("""
    <style>
    body, .stApp { background-color:#ffffff; color:#000000; }
    </style>
    """, unsafe_allow_html=True)

# =====================================================
# BUTTON UI FIX
# =====================================================
st.markdown("""
<style>
div.stButton > button {
    background:#f5f5f5 !important;
    color:#000 !important;
    border:2px solid #cfcfcf !important;
    border-radius:14px !important;
    padding:14px !important;
    font-size:18px !important;
}
div.stButton > button:hover {
    background:#e9e9e9 !important;
}
</style>
""", unsafe_allow_html=True)

# =====================================================
# LOAD MODEL
# =====================================================
@st.cache_resource
def load_assets():
    return joblib.load("final_ensemble.pkl"), joblib.load("feature_names.pkl")

model, features = load_assets()
NUM_FEATURES = len(features)

# =====================================================
# SESSION STATE
# =====================================================
if "user_state" not in st.session_state:
    st.session_state.user_state = np.zeros((1, NUM_FEATURES))
    st.session_state.asked = []
    st.session_state.done = False
    st.session_state.round_questions = 0
    st.session_state.show_checkpoint = False

# =====================================================
# HEADER
# =====================================================
st.markdown("<h1 style='text-align:center;'>🧞 Disease Akinator</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;'>Answer the questions and I will guess your disease.</p>", unsafe_allow_html=True)
st.markdown("---")

# =====================================================
# HELPERS
# =====================================================
def select_next_question():
    for i, f in enumerate(features):
        if f not in st.session_state.asked:
            return i, f
    return None, None

def update_user_state(idx, answer):
    mapping = {"Yes":1.0, "Probably":0.8, "Probably Not":0.2, "No":0.0}
    st.session_state.user_state[0, idx] = mapping[answer]

def get_probs():
    return model.predict_proba(st.session_state.user_state)[0]

# =====================================================
# QUESTION FLOW
# =====================================================
if not st.session_state.done and not st.session_state.show_checkpoint:

    q_index, question = select_next_question()

    if question is None:
        st.session_state.show_checkpoint = True
        st.rerun()

    st.markdown(f"""
    <div style="background:#f4f6fb;padding:1.5rem;border-radius:1rem;text-align:center;font-size:1.2rem;font-weight:600;">
    <span style="color:#e53935;">Are you suffering with {question}</span>?
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    answer = None
    if st.button(" Yes", use_container_width=True): answer="Yes"
    elif st.button(" Probably", use_container_width=True): answer="Probably"
    elif st.button(" Probably Not", use_container_width=True): answer="Probably Not"
    elif st.button(" No", use_container_width=True): answer="No"

    if answer:
        update_user_state(q_index, answer)
        st.session_state.asked.append(question)
        st.session_state.round_questions += 1

        probs = get_probs()
        if probs.max() >= CONFIDENCE_THRESHOLD:
            st.session_state.final_probs = probs
            st.session_state.done = True

        elif st.session_state.round_questions >= MAX_QUESTIONS:
            st.session_state.final_probs = probs
            st.session_state.show_checkpoint = True

        st.rerun()

# =====================================================
# CHECKPOINT (AFTER 25 QUESTIONS)
# =====================================================
elif st.session_state.show_checkpoint and not st.session_state.done:

    probs = st.session_state.final_probs
    idx = probs.argmax()

    st.markdown("##  Best guess so far")
    st.markdown(f"""
    <div style="background:#fff3e0;padding:1.5rem;border-radius:1rem;text-align:center;">
    <h2>{model.classes_[idx]}</h2>
    <h3>Confidence: {probs[idx]*100:.2f}%</h3>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### Are you satisfied with this answer?")

    if st.button(" Yes, this is correct", use_container_width=True):
        st.session_state.done = True
        st.session_state.show_checkpoint = False
        st.rerun()

    if st.button(" No, continue to Diagnosis", use_container_width=True):
        st.session_state.round_questions = 0
        st.session_state.show_checkpoint = False
        st.rerun()

# =====================================================
# FINAL RESULT
# =====================================================
else:
    probs = st.session_state.final_probs
    order = probs.argsort()[::-1]

    st.markdown("##  I’ve got it!")
    st.markdown(f"""
    <div style="background:#e8f5e9;padding:1.5rem;border-radius:1rem;text-align:center;">
    <h2>{model.classes_[order[0]]}</h2>
    <h3>Confidence: {probs[order[0]]*100:.2f}%</h3>
    </div>
    """, unsafe_allow_html=True)

    st.warning( "⚠️ This prediction is for educational purposes only. " "Please consult a medical professional." )

    if st.button(" Start New Diagnosis", use_container_width=True):
        st.session_state.clear()
        st.rerun()

