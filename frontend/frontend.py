import base64
import logging
import re
import uuid
from pathlib import Path

import requests
import streamlit as st

# Constants
API_URL = "http://host.docker.internal:8000/api/v1"
AVAILABLE_MODELS = [
    "gemma-3-27b-it",
    "llama-3.3-70b-instruct",
    "llama-3.1-sauerkrautlm-70b-instruct",
    "qwq-32b",
    "mistral-large-instruct",
    "qwen3-235b-a22b"
]
PATIENT_ROLES = ["default", "alzheimer", "schwerhörig", "verdrängung"]
TALKATIVENESS_LEVELS = ["kurz angebunden", "ausgewogen", "ausschweifend"]

# Setup logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def init_session_state() -> None:
    """Initialize Streamlit session state variables"""
    if "condition" not in st.session_state:
        st.session_state.model = "qwen3-235b-a22b"
        st.session_state.condition = "alzheimer"
        st.session_state.talkativeness = "kurz angebunden"
        st.session_state.session_id = str(uuid.uuid4())
    if "messages" not in st.session_state:
        st.session_state.messages = []


def load_patient_image() -> str:
    """Load and convert patient image to base64"""
    def img_to_base64(image_path: Path) -> str:
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode()

    img_path = Path(__file__).parent / "assets" / "anna_zank.png"
    return img_to_base64(img_path)

def setup_header_layout() -> None:
    """Configure header layout"""
    st.set_page_config(page_title="Symptex", page_icon="🤖")
    st.markdown("""
        <style>
            .patient-image {
                width: 90px;
                height: 90px;
                border-radius: 50%;
                object-fit: cover;
            }
            .title-container {
                display: flex;
                align-items: center;
            }
            .header-section {
                padding-bottom: 2rem;
            }
        </style>
    """, unsafe_allow_html=True)

def create_header(img_base64: str) -> None:
    """Create the header with patient image and name"""
    st.markdown('<div class="header-section">', unsafe_allow_html=True)
    col1, col2 = st.columns([0.6, 3.4])
    with col1:
        st.markdown(f"""
            <div style="text-align: center;">
                <img src="data:image/png;base64,{img_base64}" 
                     class="patient-image">
            </div>
        """, unsafe_allow_html=True)
    with col2:
        st.title("Anna Zank")
    st.markdown('</div>', unsafe_allow_html=True)

def display_patient_info() -> None:
    """Display patient information box"""
    st.info(
        """**Fallvignette**:

Notfallmäßige Vorstellung mit dem RTW bei Sturz auf die rechte Hüfte im häuslichen Umfeld vor ca. 3 Stunden. Deutliche Druckdolenz über der rechten Hüfte, pDMS intakt.

**Patientenstammdaten**:

* Alter: 89 Jahre
* Geburtsdatum: 01.09.1935
* Ethnie: kaukasisch
* BMI: 20,5"""
)

def setup_sidebar() -> None:
    """Setup sidebar controls"""
    st.sidebar.selectbox("Modell", options=AVAILABLE_MODELS, key="model")
    st.sidebar.selectbox("Patientenrolle", options=PATIENT_ROLES, key="condition")
    st.sidebar.selectbox("Gesprächsverhalten", options=TALKATIVENESS_LEVELS, key="talkativeness")

def handle_chat_reset() -> None:
    """Handle chat reset functionality"""
    try:
        # Send request to clear db for this session
        response = requests.post(f"{API_URL}/reset/{st.session_state.session_id}")
        if response.status_code == 200:
            # Generate new session ID
            st.session_state.session_id = str(uuid.uuid4())
            # Clear frontend messages
            st.session_state.messages = []
            st.rerun()
        else:
            st.error("Error resetting chat memory")
    except Exception as e:
        logger.error(f"Error resetting chat: {str(e)}")
        st.error(f"Could not reset chat memory: {str(e)}")

def handle_chat_eval() -> None:
    """Handle chat rating functionality."""
    if not st.session_state.messages:
        st.warning("Der Chat enthält noch keine Nachrichten zur Bewertung.")
        return

    try:
        messages = [
            {"role": msg["role"], "output": msg["output"]} for msg in st.session_state.messages
        ]

        # Create placeholder for evaluation response
        response_placeholder = st.chat_message("patient").markdown("")

        with st.spinner("Anamnese Feedback wird erstellt..."):
            with requests.post(f"{API_URL}/eval", json={"messages": messages}, stream=True) as response:
                if response.status_code == 200:
                    evaluation_text = process_llm_response(response, response_placeholder)
                    st.session_state.messages.append({
                        "role": "patient",
                        "output": evaluation_text,
                    })
                else:
                    st.error(f"Fehler bei der Bewertung (Status: {response.status_code})")

    except Exception as e:
        logger.error(f"Error evaluating chat: {str(e)}")
        st.error(f"Fehler bei der Bewertung: {str(e)}")

def process_llm_response(response: requests.Response, response_placeholder: st.delta_generator.DeltaGenerator) -> str:
    """Process streaming response from LLM"""
    streamed_text = ""
    buffer = ""
    think_tags_removed = False

    for chunk in response.iter_content(chunk_size=None):
        chunk_text = chunk.decode()
        buffer += chunk_text
        
        # Check for think tags
        if not think_tags_removed and "<think>" in buffer:
            # Wait for closing tag
            if "</think>\n" in buffer:
                # Remove tags
                buffer = re.sub(r'^<think>[\s\S]*?</think>\n\n?', '', buffer)
                think_tags_removed = True
            else:
                # Don't display anything yet, wait for closing tag
                continue
        
        # Add processed text to stream
        if think_tags_removed or not "<think>" in buffer:
            streamed_text += buffer
            buffer = ""
            think_tags_removed = True if not think_tags_removed else think_tags_removed

        # Update display
        response_placeholder.markdown(streamed_text)
    
    return streamed_text

def main() -> None:
    """Main application function"""
    setup_header_layout()
    img_base64 = load_patient_image()
    init_session_state()
    
    create_header(img_base64)
    display_patient_info()
    setup_sidebar()

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if "output" in message:
                st.markdown(message["output"])

    # Handle user input
    if prompt := st.chat_input("Fange hier ein Gespräch an..."):
        with st.chat_message("user"):
            st.markdown(prompt)

        st.session_state.messages.append({"role": "user", "output": prompt})

        data = {
            "message": prompt,
            "model": st.session_state.model,
            "condition": st.session_state.condition,
            "talkativeness": st.session_state.talkativeness,
            "patient_file_id": 3, # Anna Zank
            "session_id": st.session_state.session_id
        }

        with st.spinner("Denkt nach..."):
            response_placeholder = st.chat_message("assistant").markdown("")
            # todo remove this
            logger.info("Calling LLM lmao lmao yeah")
            with requests.post(API_URL + "/chat", json=data, stream=True) as response:

                if response.status_code == 200:
                    streamed_text = process_llm_response(response, response_placeholder)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "output": streamed_text,
                    })
                else:
                    st.error("An error occurred while processing your message.")

    # Add sidebar buttons
    if st.sidebar.button("Chat zurücksetzen", use_container_width=True):
        handle_chat_reset()
    if st.sidebar.button("Chat bewerten", use_container_width=True):
        handle_chat_eval()

if __name__ == "__main__":
    main()
