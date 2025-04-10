from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate


model = OllamaLLM(model = "mistral")

template = """
You are an ai agent who helps to answer the question realted to the university :
Here are  some useful information about the university: {information}
Here is the question to  answers : {questions}"""


prompt = ChatPromptTemplate.from_template(template)  #

chain = prompt | model  # this helps to create the chain to provide required data 

while True : 
 question = input()

 result = chain.invoke({"information":[],"questions":"What is the name of this university?"})

 print(result)