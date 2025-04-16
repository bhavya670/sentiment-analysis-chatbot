from flask import Flask, request, jsonify, render_template
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk

# Download VADER lexicon if not already downloaded
nltk.download('vader_lexicon')

app = Flask(__name__)
sia = SentimentIntensityAnalyzer()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    data = request.get_json()
    user_input = data.get('message', '')
    sentiment_scores = sia.polarity_scores(user_input)
    
    if sentiment_scores['compound'] >= 0.05:
        sentiment = 'Positive'
    elif sentiment_scores['compound'] <= -0.05:
        sentiment = 'Negative'
    else:
        sentiment = 'Neutral'
    
    return jsonify({'sentiment': sentiment, 'scores': sentiment_scores})

if __name__ == '__main__':
    app.run(debug=True)

# HTML code for User Interface (index.html)
html_code = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Sentiment Analysis Chatbot</title>
    <script>
        async function sendMessage() {
            let userMessage = document.getElementById("userMessage").value;
            let response = await fetch("/analyze", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ message: userMessage })
            });
            let data = await response.json();
            document.getElementById("response").innerText = "Sentiment: " + data.sentiment;
        }
    </script>
</head>
<body>
    <h1>Sentiment Analysis Chatbot</h1>
    <input type="text" id="userMessage" placeholder="Type a message...">
    <button onclick="sendMessage()">Analyze</button>
    <p id="response"></p>
</body>
</html>
"""

# Save HTML file
with open("templates/index.html", "w") as file:
    file.write(html_code)
