import streamlit as st
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import OllamaLLM
from vector import retrieve

st.set_page_config(page_title="My Chatbot", page_icon=":robot:", layout="wide")
st.title("Chat Bot")
st.write("This is a simple chatbot using LangChain and Ollama.")

if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if input_txt := st.chat_input("Shoot your questions here..."):
    with st.chat_message("user"):
        st.markdown(input_txt)
        st.session_state.chat_history.append({"role": "user", "content": input_txt})

    prompt = ChatPromptTemplate.from_messages(
        [("system","You are a helpful assistant. USe this document to answer the questions: {retrieved_info} and the question is {input}"),
        ("user","user input : {input}")]
    )

    llm = OllamaLLM(model="llama2", temperature=0.5,base_url="http://ollama:11434")
    output_parser = StrOutputParser()
    llm_chain = prompt | llm | output_parser

    
    with st.spinner("Thinking..."):
        retrieved_info = retriever.invoke(input_txt) 
        response = llm_chain.invoke({"input": input_txt})

    with st.chat_message("assistant"):
        st.markdown(response)
        st.session_state.chat_history.append({"role": "assistant", "content": response})