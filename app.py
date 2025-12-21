import streamlit as st
import numpy as np
import joblib

# =====================================================
# PAGE CONFIG
# =====================================================
st.set_page_config(
    page_title="Disease Akinator",
    page_icon="🧞",
    layout="centered"
)

# =====================================================
# THEME STATE (LIGHT BY DEFAULT)
# =====================================================
if "theme" not in st.session_state:
    st.session_state.theme = "light"

# Sidebar toggle
with st.sidebar:
    st.markdown("## 🎨 Theme")
    theme_choice = st.radio(
        "Choose mode:",
        ["Light", "Dark"],
        index=0
    )
    st.session_state.theme = theme_choice.lower()

# =====================================================
# THEME STYLES
# =====================================================
if st.session_state.theme == "dark":
    st.markdown("""
        <style>
        body, .stApp {
            background-color: #0e1117;
            color: #fafafa;
        }
        </style>
    """, unsafe_allow_html=True)
else:
    st.markdown("""
        <style>
        body, .stApp {
            background-color: #ffffff;
            color: #000000;
        }
        </style>
    """, unsafe_allow_html=True)

# =====================================================
# 🔥 BUTTON UI FIX (THIS IS THE IMPORTANT PART)
# =====================================================
st.markdown("""
<style>
/* Fix all Streamlit buttons */
div.stButton > button {
    background-color: #f5f5f5 !important;   /* light grey */
    color: #000000 !important;              /* black text */
    border: 2px solid #cfcfcf !important;   /* visible border */
    border-radius: 14px !important;
    padding: 14px !important;
    font-size: 18px !important;
    font-weight: 500 !important;
}

/* Hover effect */
div.stButton > button:hover {
    background-color: #e9e9e9 !important;
    border-color: #9e9e9e !important;
}

/* Mobile fix */
@media (max-width: 600px) {
    div.stButton > button {
        font-size: 16px !important;
        padding: 12px !important;
    }
}
</style>
""", unsafe_allow_html=True)

# =====================================================
# LOAD MODEL FILES (CACHED)
# =====================================================
@st.cache_resource
def load_assets():
    model = joblib.load("final_ensemble.pkl")
    features = joblib.load("feature_names.pkl")
    return model, features

model, features = load_assets()

NUM_FEATURES = len(features)
CONFIDENCE_THRESHOLD = 0.90

# =====================================================
# SESSION STATE
# =====================================================
if "user_state" not in st.session_state:
    st.session_state.user_state = np.zeros((1, NUM_FEATURES))
    st.session_state.asked = []
    st.session_state.done = False
    st.session_state.possible_mask = np.ones(NUM_FEATURES, dtype=bool)

# =====================================================
# HEADER
# =====================================================
st.markdown(
    "<h1 style='text-align:center;'>🧞 Disease Akinator</h1>",
    unsafe_allow_html=True
)
st.markdown(
    "<p style='text-align:center; font-size:1rem;'>"
    "Answer the questions and I will guess your disease.</p>",
    unsafe_allow_html=True
)
st.markdown("---")

# =====================================================
# HELPER FUNCTIONS
# =====================================================
def select_next_question():
    for i, f in enumerate(features):
        if f not in st.session_state.asked and st.session_state.possible_mask[i]:
            return i, f
    return None, None

def update_user_state(idx, answer):
    mapping = {
        "Yes": 1.0,
        "Probably": 0.8,
        "Probably Not": 0.2,
        "No": 0.0
    }
    st.session_state.user_state[0, idx] = mapping[answer]
    st.session_state.possible_mask[idx] = answer in ["Yes", "Probably"]

def check_prediction():
    probs = model.predict_proba(st.session_state.user_state)[0]
    return probs

# =====================================================
# MAIN FLOW
# =====================================================
if not st.session_state.done:

    q_index, question = select_next_question()

    if question is None:
        st.session_state.done = True
    else:
        st.markdown(
            f"""
            <div style="
                background:#f4f6fb;
                padding:1.5rem;
                border-radius:1rem;
                text-align:center;
                font-size:1.2rem;
                font-weight:600;
            ">
            <span style="color:#e53935;">Are you suffering with {question}</span>?
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown("<br>", unsafe_allow_html=True)

        if st.button(" Yes", use_container_width=True):
            answer = "Yes"
        elif st.button(" Probably", use_container_width=True):
            answer = "Probably"
        elif st.button(" Probably Not", use_container_width=True):
            answer = "Probably Not"
        elif st.button(" No", use_container_width=True):
            answer = "No"
        else:
            answer = None

        if answer:
            update_user_state(q_index, answer)
            st.session_state.asked.append(question)

            probs = check_prediction()
            best_idx = probs.argmax()
            best_prob = probs[best_idx]

            if best_prob >= CONFIDENCE_THRESHOLD:
                st.session_state.done = True
                st.session_state.final_probs = probs

            st.rerun()

# =====================================================
# FINAL RESULT SCREEN
# =====================================================
else:
    probs = st.session_state.final_probs
    sorted_idx = probs.argsort()[::-1]

    st.markdown("## 🎯 I’ve got it!")

    st.markdown(
        f"""
        <div style="
            background:#e8f5e9;
            padding:1.5rem;
            border-radius:1rem;
            text-align:center;
            margin-bottom:1rem;
        ">
        <h2>{model.classes_[sorted_idx[0]]}</h2>
        <h3 style="color:#2e7d32;">
            Confidence: {probs[sorted_idx[0]]*100:.2f}%
        </h3>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("### 🔎 Other Possible Matches")
    for i in sorted_idx[1:3]:
        st.markdown(
            f"""
            <div style="
                background:#f1f1f1;
                padding:1rem;
                border-radius:0.75rem;
                margin-bottom:0.5rem;
            ">
            <strong>{model.classes_[i]}</strong><br>
            Confidence: {probs[i]*100:.2f}%
            </div>
            """,
            unsafe_allow_html=True
        )

    st.warning(
        "⚠️ This prediction is for educational purposes only. "
        "Please consult a medical professional."
    )

    if st.button("🔁 Start New Diagnosis", use_container_width=True):
        st.session_state.user_state = np.zeros((1, NUM_FEATURES))
        st.session_state.asked = []
        st.session_state.done = False
        st.session_state.possible_mask = np.ones(NUM_FEATURES, dtype=bool)
        st.rerun()
