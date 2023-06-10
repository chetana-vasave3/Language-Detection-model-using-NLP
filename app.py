from flask import Flask, render_template, request
import pandas as pd
import string
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

app = Flask(__name__)

df = pd.read_csv("text.csv")

def remove_punctuations(text):
    for pun in string.punctuation:
        text = text.replace(pun, "")
    text = text.lower()
    return text

df['Cleaned_text'] = df['Text'].apply(remove_punctuations)

X = df.Cleaned_text
y = df.Language

vec = TfidfVectorizer(ngram_range=(1, 2), analyzer="char")
NLP_model = LogisticRegression()

X = vec.fit_transform(X)
NLP_model.fit(X, y)

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        text_input = request.form['text_input']
        text_input = remove_punctuations(text_input)
        text_vector = vec.transform([text_input])
        prediction = NLP_model.predict(text_vector)[0]
        return render_template('result.html', prediction=prediction)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
