import time
import tracemalloc
import streamlit as st
import sys
import os

tracemalloc.start()

# Get the absolute path to the src directory
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src'))

# Add the src directory to the sys.path
sys.path.insert(0, src_path)

from inference.LLMInference import LLMInference
from main import MainRunner
from embeddings.EmbeddingTransformer import EmbeddingTransformer

st.title("Stanford Course Review Chatbot")

# Cache the creation of LLMInference
@st.cache_resource
def get_llm_client():
    return LLMInference()

# Cache the creation of MainRunner
@st.cache_resource
def running_app_background_process():
    return MainRunner()

# Initialize objects using cached functions
llm_Client = get_llm_client()
app_runner = running_app_background_process()

# Session state initialization
if "openai_model" not in st.session_state:
    st.session_state["openai_model"] = "gpt-4o-mini"

if "messages" not in st.session_state:
    st.session_state.messages = []

# Initial topics
topics = app_runner.available_topics

# Initialize session state for pagination
if "topic_index" not in st.session_state:
    st.session_state.topic_index = 0

# Number of topics to display at a time
topics_per_page = 5

# Display topics
current_topics = topics[st.session_state.topic_index:st.session_state.topic_index + topics_per_page]
selected_topic = st.selectbox("Select a topic:", current_topics)
col1, col2 = st.columns(2)

# Button to load more topics
with col1:
    if st.button("Load next topics"):
        st.session_state.topic_index += topics_per_page
        if st.session_state.topic_index >= len(topics):
            st.session_state.topic_index = 0  # Reset to start if we reach the end

with col2:
    if st.button("Load previous topics"):
        st.session_state.topic_index -= topics_per_page
        if st.session_state.topic_index <= len(topics):
            st.session_state.topic_index = 0  # Reset to start if we reach the end

st.write(f"Selected topic: {selected_topic}")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if user_prompt := st.chat_input("What is up?"):
    st.session_state.messages.append({"role": "user", "content": f"Topic: {selected_topic}, Query: {user_prompt}"})
    with st.chat_message("user"):
        st.markdown(f"**Topic:** {selected_topic}  \n**Query:** {user_prompt}")

    assistant_placeholder = st.chat_message("assistant")
    with assistant_placeholder:
        response_placeholder = st.empty()
        response_placeholder.markdown("Thinking...")

    system_prompt_by_semantic_search = app_runner.generate_system_prompt(user_prompt)
    stream = llm_Client.generate_response(system_prompt_by_semantic_search, streaming=True)
    full_response = ""
    for chunk in llm_Client.extract_llm_streaming_response(stream):
        full_response += chunk
        response_placeholder.markdown(full_response + "_")
        time.sleep(0.02)

    st.session_state.messages.append({"role": "assistant", "content": full_response})

current, peak = tracemalloc.get_traced_memory()
print(f"Current memory usage: {current / 10**6} MB")
print(f"Peak memory usage: {peak / 10**6} MB")

tracemalloc.stop()