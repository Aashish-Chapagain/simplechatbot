from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document 
import os 
import pandas as pd 

df = pd.read_csv("campus_info.csv") #data frame 

embeddings = OllamaEmbeddings(model = "mistral")

db_location = "./chroma_langchain_db"

add_document =  not os.path.exists(db_location)        


if add_document : 
    documents = []
    ids = []

    for i, row in df.iterrows():
        document = Document(
            page_content= row["Category"] + "" + row["Subcategory"] + "" + row["Details"] + "" +  row["Link"],
            metadata = {"Subcategory": row["Subcategory"], "Details": row["Details"]},
            ids = str(i)
        )
        ids.append(str(i))
        documents.append(document)

vector_store = Chroma(
    collection_name = "campus_information",
    persist_directory = db_location,
    embedding_function= embeddings

)
if add_document :   
    vector_store.add_documents(documents = documents, ids = ids )


retriever = vector_store.as_retriever(
    search_kwargs = {"k" : 1}
)
