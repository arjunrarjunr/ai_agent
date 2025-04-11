from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
import os
import pandas as pd

df = pd.read_csv("data/faq.csv")
embeddings= OllamaEmbeddings(model="bge-small-en")

db_location = "./chroma_langchain_db"
add_documents = not os.path.exists(db_location)

if add_documents:
    documents = []
    ids = []

    for i,row in df.iterrows():
        document = Document(
        page_content= str(row["question"]),
        metadata = {"question": row["question"], "answer": row["answer"]},
        id=str(i)
        )
        ids.append(str(i))
        documents.append(document)

vectorstore = Chroma.from_documents(
    collection_name="faq",
    persist_directory=db_location,
    embedding_function=embeddings
)

if add_documents:
    vectorstore.add_documents(documents, ids=ids)

retriever = vectorstore.as_retriever(search_kwargs={"k": 2})