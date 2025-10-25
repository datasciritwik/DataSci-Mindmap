import streamlit as st
import json
import os, re
import tempfile
import streamlit.components.v1 as components
from streamlit.components.v1 import html
import time 
from gtts import gTTS
import base64
# from tts import text_to_speech

# Set up the page
st.set_page_config(
    page_title="DataSci Mindmap",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Initialize session state for TTS
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
agents = load_json(".paths/agents.json")

# Combine into one main topic dictionary
topics_dict = {
    agents['topic']:agents['path'],
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

    # 🎧 Listen button
    if st.button("🔊 Listen"):
        text = st.session_state.tts_text.strip()

        if text:
            with st.spinner("Generating audio..."):
                # Generate TTS
                # 2. Remove markdown symbols (*, _, #, >, `, etc.)
                # 3. Remove extra whitespace/newlines
                text = re.sub(r'[*_#>\-`~\[\](){}>]+', ' ', text)
                text = re.sub(r'\s+', ' ', text).strip()
                output_path = f"{int(time.time())}.wav"


                tts = gTTS(text)
                tts.save(output)
                # Create base64 for direct embedding
                with open(output_path, "rb") as f:
                    audio_bytes = f.read()
                b64 = base64.b64encode(audio_bytes).decode()

                # Audio player
                # st.audio(audio_bytes, format="audio/mp3")
                # st.audio(audio_bytes, format="audio/mpeg")
                st.audio(audio_bytes, format="audio/wav")
            # Fallback download link for iPhone users
            # st.markdown(
            #     f'<a href="data:audio/mp3;base64,{b64}" download="speech.mp3">📥 Download / Play Audio (iPhone Friendly)</a>',
            #     unsafe_allow_html=True
            # )
            # st.markdown(
            #     f'<a href="data:audio/mpeg;base64,{b64}" download="speech.mpeg">📥 Download / Play Audio (iPhone Friendly)</a>',
            #     unsafe_allow_html=True
            # )
            

        else:
            st.warning("No content to read.")
else:
    st.error(f"⚠️ File not found: `{file_path}`")

# Back to Top button
st.markdown(''' 
<a target="_self" href="#interview-prep">
    <button>
        Back to Top
    </button>
</a>
''', unsafe_allow_html=True)

# Hide Streamlit footer
html('''
<script>
window.top.document.querySelectorAll(`[href*="streamlit.io"]`).forEach(e => e.setAttribute("style", "display: none;"));
</script>
''')

