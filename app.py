import streamlit as st
from transformers import pipeline
import google.generativeai as genai
import time

API_KEY = "PASTE_KEY_HERE"

try:
    genai.configure(api_key=API_KEY)
except Exception as e:
    st.error(f"Error configuring API: {e}")

@st.cache_resource
def load_intent_model():
   
    with st.spinner("Loading Intent Detection Model (this happens only once)..."):
        
        return pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

def detect_intent(text):

    classifier = load_intent_model()
    
    candidate_labels = ["Academic/Educational", "Entertainment/Fun", "Technical/Code", "Personal/Casual", "General Info"]
    
    result = classifier(text, candidate_labels)

    top_intent = result['labels'][0]
    confidence = result['scores'][0]
    
    return top_intent, confidence

def generate_styled_response(query, intent, style):
   
    system_prompt = f"""
    ROLE: You are an AI acting as a specific persona.
    
    CONTEXT:
    - User's Query: "{query}"
    - Detected Intent of Query: "{intent}"
    
    YOUR PERSONA: "{style}"
    
    GUIDELINES:
    1. Answer the query accurately. Do not hallucinate facts.
    2. STRICTLY adopt the tone, vocabulary, and mannerisms of the persona.
    3. IF 'Overconfident Genius': Be condescending, use big words, act superior.
    4. IF 'Nervous Intern': Stutter (um, uh), apologize profusely, sound unsure but correct.
    5. IF 'Sarcastic Reviewer': Be dry, critical, and act like this is a waste of time.
    6. IF 'Pirate': Speak entirely in pirate slang.
    """
    
    models_to_try = ['gemini-2.0-flash', 'gemini-2.5-flash', 'gemini-flash-latest']
    
    last_error = ""
    
    for model_name in models_to_try:
        try:
            model = genai.GenerativeModel(model_name)
            response = model.generate_content(system_prompt)
            return response.text
        except Exception as e:
            last_error = str(e)
            continue 
            
    try:
        available_models = []
        for m in genai.list_models():
            if 'generateContent' in m.supported_generation_methods:
                available_models.append(m.name)
        
        return f"""
        ⚠️ **All Model Attempts Failed.**
        
        **Last Error:** {last_error}
        
        **Available Models for your Key:**
        {", ".join(available_models)}
        
        *Please look at the list above. If you see a name like 'models/gemini-pro', you might need to update the code to use that exact name.*
        """
    except Exception as e_debug:
        return f"Critical Error: Could not even list models. Check your API Key. ({e_debug})"


st.set_page_config(page_title="Vinnovate AI Project", page_icon="🚀", layout="centered")

# Header
st.title("🚀 Intent-Aware & Style-Adaptive System")
st.markdown("### Built for VinnovateIT Recruitment (Round 2)")
st.markdown("---")

# Sidebar
st.sidebar.header("⚙️ Configuration")
st.sidebar.info("This tool detects the *intent* of your query using a BERT model, then answers using a Generative model in a specific *style*.")

selected_style = st.sidebar.selectbox(
    "Choose Response Persona:",
    ["Overconfident Genius", "Nervous Intern", "Sarcastic Reviewer", "Calm Professor", "Pirate"]
)

# Main Interaction Area
query = st.text_area("Enter your query:", height=100, placeholder="e.g., Explain Quantum Physics to me.")

col1, col2 = st.columns([1, 4])

with col1:
    generate_btn = st.button("🚀 Run AI", type="primary")

if generate_btn:
    if not query.strip():
        st.warning("Please enter a query first.")
    else:
        # 1. Intent Detection Phase
        st.subheader("1️⃣ Intent Analysis")
        start_time = time.time()
        
        # We run the intent detection
        intent, score = detect_intent(query)
        
        # Display Intent Metrics
        st.success(f"**Detected Intent:** {intent}")
        st.progress(score, text=f"Confidence Score: {score:.2f}")
        
        # 2. Response Generation Phase
        st.subheader(f"2️⃣ Response ({selected_style})")
        with st.spinner(f"Consulting the {selected_style}..."):
            response_text = generate_styled_response(query, intent, selected_style)
            
        # Display Result card
        st.info(response_text)
        
        end_time = time.time()
        st.caption(f"Total processing time: {end_time - start_time:.2f} seconds")

# Footer
st.markdown("---")
st.markdown("<center>Made using Streamlit, HuggingFace & Gemini</center>", unsafe_allow_html=True)