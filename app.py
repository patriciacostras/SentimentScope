from flask import Flask, request, jsonify, render_template
from transformers import pipeline
import os

# Inițializează aplicația Flask
app = Flask(__name__)

# Verifică dacă modelul există
if not os.path.exists('./my_sentiment_model'):
    raise FileNotFoundError("Modelul nu a fost găsit. Asigură-te că ai antrenat și salvat modelul înainte de a rula această aplicație.")

# Încarcă modelul de analiză a sentimentelor
sentiment_pipeline = pipeline("sentiment-analysis", model='./my_sentiment_model')

# Ruta pentru pagina principală
@app.route('/')
def home():
    return render_template('index.html')

# Ruta pentru analiza sentimentelor
@app.route('/analyze', methods=['POST'])
def analyze():
    text = request.form.get('text')
    if not text:
        return render_template('index.html', error="Te rog să introduci un text pentru analiză.")
    
    result = sentiment_pipeline(text)
    return render_template('index.html', result=result[0], text=text)

# Ruta API pentru predictie
@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    if not data or 'text' not in data:
        return jsonify({"error": "Te rog să furnizezi un text în corpul cererii."}), 400
    
    text = data['text']
    result = sentiment_pipeline(text)
    return jsonify(result)

# Pornirea aplicației
if __name__ == '__main__':
    app.run(debug=True)