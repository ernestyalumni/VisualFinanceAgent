import streamlit as st
import random

# Set page title and favicon
st.set_page_config(page_title="Simple Chatbot", page_icon="🤖")

# Initialize session state variables
if 'generated' not in st.session_state:
    st.session_state['generated'] = []
if 'past' not in st.session_state:
    st.session_state['past'] = []
if 'messages' not in st.session_state:
    st.session_state['messages'] = [
        {"role": "system", "content": "You are a helpful assistant."}
    ]

# Sidebar for app title, description, and sample PDF summaries
st.sidebar.title("Simple Chatbot")
st.sidebar.markdown("This is a basic chatbot app built with Streamlit.")

# Sample PDF summaries
st.sidebar.header("Sample PDF Summaries")
sample_summaries = {
    "Research Paper 1": "This paper discusses the impact of climate change on marine ecosystems...",
    "Technical Report": "The report outlines the latest advancements in quantum computing...",
    "Case Study": "This case study examines the successful implementation of AI in healthcare...",
}

for title, summary in sample_summaries.items():
    with st.sidebar.expander(title):
        st.write(summary)

# Function to generate response (placeholder)
def generate_response(prompt):
    responses = [
        "That's interesting! Tell me more.",
        "I see. How does that make you feel?",
        "Fascinating! What do you think about that?",
        "I understand. Can you elaborate on that?",
        "That's a great point. What else comes to mind?",
    ]
    return random.choice(responses)

# Chat interface
st.title("Chat with the Bot")

# User input
user_input = st.text_input("You:", key="user_input")

if user_input:
    # Add user message to chat history
    st.session_state['past'].append(user_input)
    st.session_state['messages'].append({"role": "user", "content": user_input})

    # Generate response
    response = generate_response(user_input)
    
    # Add bot response to chat history
    st.session_state['generated'].append(response)
    st.session_state['messages'].append({"role": "assistant", "content": response})

# Display chat history
chat_placeholder = st.empty()
with chat_placeholder.container():
    for i in range(len(st.session_state['generated'])-1, -1, -1):
        st.markdown(f"**Bot:** {st.session_state['generated'][i]}")
        st.markdown(f"**You:** {st.session_state['past'][i]}")

# Run the app: streamlit run streamlit/main.py