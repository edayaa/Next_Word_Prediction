from flask import Flask, request, render_template
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import joblib
import pickle
import numpy as np
import string

app = Flask(__name__)
#model = joblib.load("next_word_1210.pkl")
model = joblib.load('word_pred_1222.pkl')
with open('tokenizer_1222.pkl', 'rb') as handle:
    tokenizer = pickle.load(handle)
print("after tokenizer load")
    
@app.route("/")
def hello():
    return render_template('index.html')

@app.route('/predict',methods = ['POST'])
def result():
    
    if request.method == 'POST':
        in_text = request.form["feedtext"]
        noofwords  = request.form["noofwords"]
        if in_text.strip() == "":
            print("in_text",in_text)
            return render_template('index.html', prediction_text="Predicted Text :  {} ".format("Enter Your Input text"))
        if noofwords.strip() == "":
            print("noofwords",noofwords)
            return render_template('index.html', prediction_text="Predicted Text :  {} ".format("Enter No of Words to predict"))
        no_of_words = int(noofwords)
        print("no_of_words",no_of_words)
        for _ in range(no_of_words):
            encoded = tokenizer.texts_to_sequences([in_text])[0] 
            print("encoded : ", encoded)
            encoded = pad_sequences([encoded], maxlen=99, padding='pre')
            yhat = model.predict_classes(encoded, verbose=0)
            out_word = ''
            for word, index in tokenizer.word_index.items():
                if index == yhat:
                    out_word = word
                    break                  
            in_text += ' ' + out_word                
       

        return render_template('index.html', prediction_text="Predicted Text :  {} ".format(in_text))
    
if __name__ == "__main__":
    app.run()  