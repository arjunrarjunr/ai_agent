import streamlit as st
import pandas as pd
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import OllamaLLM
import os

def vectorising_the_data(df):
    # Initialize embeddings
    embeddings = OllamaEmbeddings(model="mxbai-embed-large", base_url="http://ollama:11434")

    db_location = "./credit_card_statement_db_test"
    add_documents = not os.path.exists(db_location)

    documents = []
    ids = []

    if add_documents:
        # Create new documents if the database does not exist
        for i, row in df.iterrows():
            document = Document(
                page_content=f"Date: {row['Date']}, Merchant Category: {row['Merchant category']}, Amount: {row['Amount']}, Details: {row['Transaction details']}",
                metadata={"Date": row["Date"], "Merchant category": row["Merchant category"], "Amount": row["Amount"], "Details": row["Transaction details"]},
                id=str(i)
            )
            ids.append(str(i))
            documents.append(document)

        # Create a new vectorstore and persist it
        vectorstore = Chroma.from_documents(
            documents=documents,
            embedding=embeddings,
            collection_name="credit_card_statement",
            persist_directory=db_location
        )
        vectorstore.add_documents(documents=documents, ids=ids)
    else:
        # Load the existing vectorstore
        vectorstore = Chroma(
            persist_directory=db_location,
            embedding_function=embeddings,
            collection_name="credit_card_statement"
        )

    # Create a retriever from the vectorstore
    retriever = vectorstore.as_retriever(search_kwargs={"k" : 5}) 

    return retriever

def main() :
    st.set_page_config(page_title="Credit Card Statement", page_icon=":💳:", layout="wide")
    st.title("Credit Card Statement Analysis")
    st.write("Upload your credit card statement in CSV format.")

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

        pre_crafted_prompts = [
            "What is my Monthly Spending Trends ?",
            "What is my spend Pattern by Merchants?",
            "List down my High value transactions above Rs.3000"
        ]

        # Dropdown for selecting a pre-crafted prompt
        selected_prompt = st.selectbox("Choose a pre-crafted question:", [""] + pre_crafted_prompts)

        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = []

        for message in st.session_state.chat_history:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        
        input_txt = st.chat_input("Shoot your questions here...") or selected_prompt

        if input_txt:
            with st.chat_message("user"):
                st.markdown(input_txt)
                st.session_state.chat_history.append({"role": "user", "content": input_txt})

            prompt = ChatPromptTemplate.from_messages(
                [("system",'''You are an AI assistant helping the user to analyze the credit card statement data.
                  Please use this as the statement data {statement} and answer the question {input}.''')]
            )

            llm = OllamaLLM(model="llama3.2",base_url="http://ollama:11434")
            output_parser = StrOutputParser()
            llm_chain = prompt | llm | output_parser

            
            with st.spinner("Thinking..."):
                retriever = vectorising_the_data(df) 
                relevant_documents = retriever.get_relevant_documents(input_txt)

                # Format the retrieved documents into a readable string
                statement = "\n".join([doc.page_content for doc in relevant_documents])

                print(statement)  # Debugging: Print the retrieved documents
                response = llm_chain.invoke({"statement": statement, "input": input_txt})

            with st.chat_message("assistant"):
                st.markdown(response)
                st.session_state.chat_history.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    main()