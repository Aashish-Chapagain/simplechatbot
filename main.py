# main.py
from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from vector import retriever  # imports your fixed retriever

# Initialize the Ollama model
model = OllamaLLM(model="mistral")

# Prompt template
template = '''
You are an intelligent assistant with access to the following context:

{information}

Based on the above information, answer the following question clearly and accurately:
{question}
'''

prompt = ChatPromptTemplate.from_template(template)

# Combine prompt and model into a chain
chain = prompt | model

# Chat loop
while True:
    question = input("Ask your question (or 'q' to quit): ").strip().lower()
    if question.lower() == "q":
        break

    # Retrieve relevant documents
    documents = retriever.invoke(question)

    # Extract only the page content from documents
    information = "\n".join([doc.page_content for doc in documents])

    # Invoke the chain with context and question

    print("Thinking....")

    result = chain.invoke({
        "information": information,
        "question": question
    })

    print("\nAnswer:", result, "\n")
