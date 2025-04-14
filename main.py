from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from vector import retriever 


model = OllamaLLM(model = "mistral")

template = """
You are an ai agent who helps to answer the question realted to the collage :
Here are  some useful information about the university: {information}
Here is the question to  answers : {question}"""


prompt = ChatPromptTemplate.from_template(template)  

chain = prompt | model  # this helps to create the chain to provide required data 

while True : 
 question = input().strip().lower()
 if question == "q":
  break 
 information = retriever.invoke(question)
 result = chain.invoke({"information": information,"questions":"What is the name of this university?"})

 print(result)
