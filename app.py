from flask import Flask, request, jsonify, render_template
from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from vector import retriever  

app = Flask(__name__)


model = OllamaLLM(model="mistral")

template = '''
this is the information you have to read:

{information}

Based on the above information, answer the following question clearly and accurately:
{question}
'''
prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model



@app.route("/ask", methods=["POST"])
def ask_question():
    data = request.json
    question = data.get("question", "").strip()

    if not question:
        return jsonify({"error": "Missing question"}), 400

    try:
        # Retrieve documents
        documents = retriever.invoke(question)
        information = "\n".join([doc.page_content for doc in documents])

        # Invoke model chain
        result = chain.invoke({
            "information": information,
            "question": question
        })

        return jsonify({"answer": result})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


if __name__ == "__main__":
    app.run(debug=True)

