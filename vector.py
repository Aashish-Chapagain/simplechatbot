# vector.py
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
import os
import pandas as pd

# Load the data
df = pd.read_csv("campus_info.csv")

df['Category'] = df['Category'].str.lower()
df['Subcategory'] = df['Subcategory'].str.lower()
df['Link'] = df['Link'].str.lower()
df['campus name'] = df['campus name'].str.lower()

print(df['Category'],df['Subcategory'],df['Link'],df['campus name'])

# Embedding function using Ollama
embeddings = OllamaEmbeddings(model="mistral")

# Vector store directory
db_location = "./chroma_langchain_db"
add_document = not os.path.exists(db_location)

if add_document:
    documents = []
    ids = []
    for i, row in df.iterrows():
     content = f"""Campus: {row["campus name"]}
Category: {row["Category"]}
Subcategory: {row["Subcategory"]}
Details: {row["Details"]}
Link: {row["Link"] if pd.notna(row["Link"]) else "None"}
"""

     doc = Document(
        page_content=content,
        metadata={
            "Campus": row["campus name"],
            "Subcategory": row["Subcategory"],
            "Category": row["Category"]
        },
        id=str(i)
    )
    documents.append(doc)


    vector_store = Chroma(
        collection_name="campus_information",
        persist_directory=db_location,
        embedding_function=embeddings
    )

    vector_store.add_documents(documents=documents, ids=ids)

# Load the vector store
vector_store = Chroma(
    collection_name="campus_information",
    persist_directory=db_location,
    embedding_function=embeddings
)

# Retriever exposed
retriever = vector_store.as_retriever(search_kwargs={"k": 5})
