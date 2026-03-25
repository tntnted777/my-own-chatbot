import streamlit as st
from groq import Groq

st.title("QXQ's ChatBot")

api_key = "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"
client = Groq(api_key=api_key)

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("想問點什麼呢？"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile", 
            messages=st.session_state.messages,
            temperature=0.7,
            max_tokens=1024,
        )
        response = completion.choices[0].message.content
        st.markdown(response)

    st.session_state.messages.append({"role": "assistant", "content": response})