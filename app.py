from flask import Flask, request, jsonify, render_template
import requests


app = Flask(__name__)

OLLAMA_API_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "mistral" 

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.json['message']
    
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": user_input,
        "stream": False
    }

    try:
        response = requests.post(OLLAMA_API_URL, json=payload)
        data = response.json()
        return jsonify({"reply": data['response']})
    except Exception as e:
        return jsonify({"reply": f"Error: {str(e)}"})

if __name__ == '__main__':
    app.run(debug=True)
