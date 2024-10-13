import streamlit as st
import os
from streamlit_chat import message

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain_core.messages import SystemMessage

os.environ['GOOGLE_API_KEY'] = st.secrets['GOOGLE_API_KEY']

# Initialize session state variables
if 'buffer_memory' not in st.session_state:
    st.session_state.buffer_memory = ConversationBufferWindowMemory(k=3,return_messages = True)

if 'messages' not in st.session_state.keys():
    st.session_state.messages = [
        {"role":"assistant","content":"How can i help you today?"}
    ]

gemini_model = ChatGoogleGenerativeAI(model = "gemini-1.5-flash-latest")

conversation = ConversationChain(llm = gemini_model,memory = st.session_state.buffer_memory)
conversation.memory.chat_memory.add_message(SystemMessage(content = """
Your name is Gem who is a ChatBot. Your Creator is Bhuman, a 4th year student at BITS Pilani.

You were not made at Google but were made at Bhuman's Home

You explain the answers in a very funny way, using unusual examples.

Also never start the messages with like AI: or Human:
"""))
# Creating USER INTERFACE NOW

st.title("🌟 ChatBot")
st.subheader("🤖 Simple Chat Interface for an LLM by Bhuman")

# Prompt for user input and save to chat history
if prompt := st.chat_input("Your Question"):
    st.session_state.messages.append({"role":"user","content":prompt})

# Display the prior chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# If the last message is not from assistant, generate a new response
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = conversation.predict(input = prompt)
            st.write(response)
            message = {"role":"assistant","content":response}
            st.session_state.messages.append(message)
