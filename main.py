import streamlit as st
import json
import os
from streamlit.components.v1 import html

# Set up the page
st.set_page_config(
    page_title="DataSci Mindmap",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown('## Interview Prep')

# Initialize session state
if "tts_text" not in st.session_state:
    st.session_state.tts_text = ""

# Load JSON data
def load_json(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

# Load topic dictionaries
loss_data = load_json('.paths/loss.json')
regression_data = load_json('.paths/regression.json')
classification_data = load_json(".paths/classification.json")
unsupervised_data = load_json(".paths/unsupervised.json")

# Combine them into one main topic dictionary
topics_dict = {
    classification_data['topic']: classification_data['path'],
    regression_data['topic']: regression_data['path'],
    unsupervised_data['topic']: unsupervised_data['path'],
    loss_data['topic']: loss_data['path'],
}

# List of topics
all_topics = list(topics_dict.keys())

# Streamlit UI
with st.container(border=True):
    c1, c2 = st.columns(2)
    selected_topic = c1.selectbox("Select Topic", all_topics, placeholder="Select Topic", label_visibility="collapsed")
    model_names = list(topics_dict[selected_topic].keys())
    selected_model = c2.selectbox("Select Models", model_names, placeholder="Select Model", label_visibility="collapsed")

# Dynamic file loading
file_path = topics_dict[selected_topic][selected_model]

if os.path.exists(file_path):
    with open(file_path, 'r') as f:
        content = f.read()
    st.markdown(content, unsafe_allow_html=True)

    st.session_state.tts_text = content

    # 🔊 Browser-based TTS
    if st.button("🔊 Listen"):
        text = st.session_state.tts_text.replace("'", "\\'").replace("\n", " ")
        html(f"""
        <script>
            const synth = window.speechSynthesis;
            const utter = new SpeechSynthesisUtterance('{text}');
            utter.rate = 1.0;
            utter.pitch = 1.0;
            utter.lang = 'en-US';
            synth.cancel();  // stop any ongoing speech
            synth.speak(utter);
        </script>
        """, height=0)
else:
    st.error(f"⚠️ File not found: `{file_path}`")

# Back to top button
st.markdown(''' 
<a target="_self" href="#interview-prep">
    <button>Back to Top</button>
</a>
''', unsafe_allow_html=True)

# Hide Streamlit branding
html('''
<script>
window.top.document.querySelectorAll(`[href*="streamlit.io"]`)
    .forEach(e => e.setAttribute("style", "display: none;"));
</script>
''')
