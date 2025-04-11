import streamlit as st
import pandas as pd
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import OllamaLLM

st.set_page_config(page_title="Credit Card Statement", page_icon=":💳:", layout="wide")
st.title("Credit Card Statement Analysis")
st.write("Upload your credit card statement in CSV format.")

# Create two columns for splitting the screen
data_upload, chatbot  = st.columns([2,1])

with data_upload:
    st.header("Data Upload")
    st.write("Upload your credit card statement in CSV format.")
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write("Data Preview:")
        st.dataframe(df.head())

with chatbot:
    st.title("Chat Bot")
    st.write("Use this Chatbot to ask questions about your credit card statement.")

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
            [("system","You are a helpful assistant.Your question is {input}"),
            ("user","user input : {input}")]
        )

        llm = OllamaLLM(model="llama2", temperature=0.5,base_url="http://ollama:11434")
        output_parser = StrOutputParser()
        llm_chain = prompt | llm | output_parser

        
        with st.spinner("Thinking..."):
            response = llm_chain.invoke({"input": input_txt})

        with st.chat_message("assistant"):
            st.markdown(response)
            st.session_state.chat_history.append({"role": "assistant", "content": response})