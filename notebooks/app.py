import streamlit as st
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import OllamaLLM

st.set_page_config(page_title="My Chatbot", page_icon=":robot:", layout="wide")
st.title("My Chatbot")
st.write("This is a simple chatbot using LangChain and Ollama.")

input_text = st.text_input("Kindly enter your queries:")

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    ("user", "{input}"),
])  

llm = OllamaLLM(model="llama3.2", temperature=0.5, base_url="http://ollama:11434")

output_parser = StrOutputParser()

llm_chain = prompt | llm | output_parser

if input_text:
    with st.spinner("Generating response..."):
        response = llm_chain.invoke({"input": input_text})
        st.success(response)