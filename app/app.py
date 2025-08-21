from flask import Flask, request, render_template_string
import pickle

app = Flask(__name__)

with open('vectorizer.pkl', 'rb') as vf, open('model.pkl', 'rb') as mf:
    vectorizer, clf = pickle.load(vf), pickle.load(mf)

TEMPLATE = """
<!DOCTYPE html>
<title>Sentiment Analyzer</title>
<h2>Enter text for sentiment analysis</h2>
<form method=post>
  <textarea name=text rows=5 cols=40>{{ text }}</textarea><br>
  <input type=submit value=Analyze>
</form>
{% if prediction %}
<h3>Predicted Sentiment: {{ prediction }}</h3>
{% endif %}
"""

@app.route("/", methods=["GET", "POST"])
def home():
    text, prediction = "", None
    if request.method == "POST":
        text = request.form['text']
        X = vectorizer.transform([text])
        prediction = clf.predict(X)[0]
    return render_template_string(TEMPLATE, text=text, prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)