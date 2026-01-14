import streamlit as st
import torch
import plotly.graph_objects as go
from src.generator import CausalGenerator
from src.metrics import compute_hsb
from src.perturb import Perturber

# Page Config
st.set_page_config(page_title="CausalRAG Inspector", layout="wide")

# Cached Loaders (so we don't reload the model on every click)
@st.cache_resource
def load_models():
    gen = CausalGenerator(model_name="gpt2") # Switch to "meta-llama/Llama-2-7b-chat-hf" later
    attacker = Perturber()
    return gen, attacker

gen, attacker = load_models()

# --- SIDEBAR ---
st.sidebar.title("Configuration")
model_name = st.sidebar.selectbox("Model", ["gpt2", "Llama-3-8B (Mock)", "Mistral-7B (Mock)"])
temp = st.sidebar.slider("Temperature", 0.0, 1.0, 0.7)

# --- MAIN PAGE ---
st.title("🕵️ CausalRAG: Hallucination Inspector")
st.markdown("Intervention-based evaluation of RAG stability.")

col1, col2 = st.columns(2)

with col1:
    st.subheader("1. The Scenario")
    query = st.text_input("Question (q)", "Where is the Eiffel Tower?")
    evidence = st.text_area("Original Evidence (E)", "The Eiffel Tower is located in Paris, France.")
    
    if st.button("Generate Baseline Answer (y)"):
        with st.spinner("Generating..."):
            prompt = f"Context: {evidence}\nQuestion: {query}"
            answer = gen.generate(prompt)
            st.session_state['answer'] = answer
            st.success("Baseline Generated!")

    if 'answer' in st.session_state:
        st.info(f"**Model Answer (y):**\n\n{st.session_state['answer']}")

with col2:
    st.subheader("2. The Intervention")
    
    # Auto-Perturb Button
    if st.button("🎲 Auto-Perturb Evidence"):
        new_ev = attacker.perturb(evidence, strategy="adversarial")
        st.session_state['evidence_prime'] = new_ev
    
    evidence_prime = st.text_area(
        "Counterfactual Evidence (E')", 
        value=st.session_state.get('evidence_prime', "The Eiffel Tower is located in Tokyo, Japan.")
    )

    if 'answer' in st.session_state:
        if st.button("🔥 Measure Sensitivity (HSB)"):
            with st.spinner("Calculating KL Divergence..."):
                # 1. Get Logits for Original
                logits_E = gen.get_logits(evidence, st.session_state['answer'])
                
                # 2. Get Logits for Counterfactual
                logits_E_prime = gen.get_logits(evidence_prime, st.session_state['answer'])
                
                # 3. Compute Metric
                hsb = compute_hsb(logits_E, logits_E_prime)
                
                # Visualization
                st.metric(label="Hallucination Sensitivity Bound (HSB)", value=f"{hsb:.4f}")
                
                if hsb > 1.0:
                    st.success("High Sensitivity: Model respected the evidence change.")
                elif hsb > 0.1:
                    st.warning("Moderate Sensitivity: Model is uncertain.")
                else:
                    st.error("Low Sensitivity: Hallucination Risk! Model ignored evidence.")

                # Simple Gauge Chart
                fig = go.Figure(go.Indicator(
                    mode = "gauge+number",
                    value = hsb,
                    domain = {'x': [0, 1], 'y': [0, 1]},
                    title = {'text': "Causal Influence"},
                    gauge = {'axis': {'range': [None, 10]},
                             'bar': {'color': "darkblue"},
                             'steps': [
                                 {'range': [0, 0.5], 'color': "red"},
                                 {'range': [0.5, 2], 'color': "yellow"},
                                 {'range': [2, 10], 'color': "green"}],
                             }))
                st.plotly_chart(fig, use_container_width=True)

st.markdown("---")
st.caption("CausalRAG v0.1 | Master's Thesis Project")