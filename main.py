import streamlit as st
import json
import os
import streamlit.components.v1 as components
from streamlit.components.v1 import html

# Set up the page
st.set_page_config(
    page_title="DataSci Mindmap",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown('## Interview Prep')

# Load JSON data
def load_json(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

# Load topic dictionaries
loss_data = load_json('.paths/loss.json')
regression_data = load_json('.paths/regression.json')
classication_data = load_json(".paths/classification.json")
unsrpervised_data = load_json(".paths/unsupervised.json")

# Combine them into one main topic dictionary
topics_dict = {
    classication_data['topic']: classication_data['path'],
    regression_data['topic']: regression_data['path'],
    unsrpervised_data['topic']: unsrpervised_data['path'],
    loss_data['topic']: loss_data['path'],
}

# List of topics (e.g., "Loss Function", "Regression Models")
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
else:
    st.error(f"⚠️ File not found: `{file_path}`")


# As an html button (needs styling added)
st.markdown(''' <a target="_self" href="#interview-prep">
                    <button>
                        Back to Top
                    </button>
                </a>''', unsafe_allow_html=True)
                

    html('''
<script>window.top.document.querySelectorAll(`[href*="streamlit.io"]`).forEach(e => e.setAttribute("style", "display: none;"));
      </script>
    ''')