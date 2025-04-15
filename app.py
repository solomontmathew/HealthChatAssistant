import os
import warnings
import sys
import streamlit as st
import unidecode
from langchain.memory import ConversationBufferMemory
from main import create_combined_agent
import json
from langchain_core.messages import HumanMessage, AIMessage
st.set_page_config(page_title="Hospital Care")
from main import show_table_preview


tab1, tab2 = st.tabs(["Chat Assistant", "Explore Hospital Data"])


warnings.filterwarnings("ignore")
# Get the directory of the current file
current_dir = os.path.dirname(os.path.abspath(__file__))

# Define the path relative to the current file
# For example, if the directory to add is the parent directory of the current file
parent_dir = os.path.join(current_dir, "..")

# Add the parent directory to sys.path
sys.path.insert(0, parent_dir)

st.markdown("""
    <style>
        .main {background-color: #1e1e1e; color: #f1f1f1;}
        .stButton>button {border-radius: 8px; background-color: #f63366; color: white; padding: 10px 16px;}
        .stChatInputContainer input {background-color: #262730 !important; color: white !important;}
    </style>
""", unsafe_allow_html=True)



def save_messages_to_file(messages, filename="messages.json"):
    with open(filename, "w") as f:
        json.dump(messages, f)

# Load messages
def load_messages_from_file(filename="messages.json"):
    if os.path.exists(filename):
        with open(filename, "r") as f:
            return json.load(f)
    return []

def sync_session_messages_to_memory():
    messages = []
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            messages.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "assistant":
            messages.append(AIMessage(content=msg["content"]))
    st.session_state.memory.chat_memory.messages = messages

def generate_response(input_text):
    """
    Generates a response based on the given input text using the agent.

    Args:
        input_text (str): The input text to generate a response for.

    Returns:
        str: The generated response.
    """
    prompt = unidecode.unidecode(input_text)
    response = st.session_state.agent.invoke({"input": prompt})
    response_text = response["output"]
    return response_text


def reset_conversation():
    st.session_state.messages = []
    st.session_state.memory = ConversationBufferMemory(memory_key="history", input_key="input", return_messages=True)
    st.session_state.agent = create_combined_agent(memory=st.session_state.memory)



prompt = st.chat_input("Please ask your question")

with tab1:
    if "messages" not in st.session_state:
        st.session_state.messages = load_messages_from_file()

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if "memory" not in st.session_state:
        st.session_state.memory = ConversationBufferMemory(
            memory_key="history", input_key="input", return_messages=True
        )

    msg_objs = []
    for m in st.session_state.messages:
        if m["role"] == "user":
            msg_objs.append(HumanMessage(content=m["content"]))
        elif m["role"] == "assistant":
            msg_objs.append(AIMessage(content=m["content"]))
    st.session_state.memory.chat_memory.messages = msg_objs

    if "agent" not in st.session_state:
        st.session_state.agent = create_combined_agent(memory=st.session_state.memory)

    with st.sidebar:
        st.title("Hospital Care QA :rotating_light: :health_worker: :flag-us:")
        st.markdown("A Medicare/Medicaid Assistant")
        st.button("Reset Chat", on_click=reset_conversation)
        st.download_button("Download Chat Log", json.dumps(st.session_state.get("messages", [])), "chat_log.json")
        st.markdown("---")

    if prompt:
        with st.chat_message("user"):
            st.markdown(prompt)

        st.session_state.messages.append({"role": "user", "content": prompt})
        response = generate_response(prompt)

        st.session_state.memory.save_context({"input": prompt}, {"output": response})
        with st.chat_message("assistant"):
            st.markdown(response)

        st.session_state.messages.append({"role": "assistant", "content": response})
        sync_session_messages_to_memory()
        save_messages_to_file(st.session_state.messages)

with tab2:
    show_table_preview()