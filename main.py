import streamlit as st
import json
import os
import streamlit.components.v1 as components

# Set up the page
st.set_page_config(
    page_title="DataSci Mindmap",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Load JSON data
def load_json(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

# Load topic dictionaries
loss_data = load_json('.paths/loss.json')
regression_data = load_json('.paths/regression.json')

# Combine them into one main topic dictionary
topics_dict = {
    loss_data['topic']: loss_data['path'],
    regression_data['topic']: regression_data['path']
}

# List of topics (e.g., "Loss Function", "Regression Models")
all_topics = list(topics_dict.keys())

# Streamlit UI
st.markdown("<div id='top'></div>", unsafe_allow_html=True)  # Anchor for top scroll

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


# Add space before the button for better layout
st.markdown("<br><br>", unsafe_allow_html=True)

# Scroll-to-top button using JavaScript and HTML injection
components.html(
    """
    <style>
    #scrollToTopBtn {
        position: fixed;
        bottom: 40px;
        right: 40px;
        z-index: 99;
        background-color: #4CAF50;
        color: white;
        border: none;
        padding: 12px 18px;
        border-radius: 10px;
        font-size: 16px;
        cursor: pointer;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.3);
    }
    </style>

    <button onclick="topFunction()" id="scrollToTopBtn" title="Go to top">⬆️ Back to Top</button>

    <script>
    // When the user clicks the button, scroll to the top of the page
    function topFunction() {
        window.scrollTo({top: 0, behavior: 'smooth'});
    }
    </script>
    """,
    height=0,
)