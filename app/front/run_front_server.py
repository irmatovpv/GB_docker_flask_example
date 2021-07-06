import json

from flask import Flask, render_template, redirect, url_for, request
from flask_wtf import FlaskForm
from requests.exceptions import ConnectionError
from wtforms import IntegerField, SelectField, StringField
from wtforms.validators import DataRequired

from urllib.parse import unquote
import urllib.request
import json

class ClientDataForm(FlaskForm):
    title = StringField('Movie title', validators=[DataRequired()], description='Ex: "The Prestige", "Memento", "The French Connection", "Interstellar", "Thursday", "The Dark Knight Rises", "The Dark Knight", "Batman Begins", "Inception", "Police Story 3: Supercop", "El Mariachi"')
    user_id = StringField('User id', validators=[], description='Ex: 1 ')


app = Flask(__name__)
app.config.update(
    CSRF_ENABLED=True,
    SECRET_KEY='you-will-never-guess',
)

def get_prediction(title, user_id):
    body = {'title': title,
                            'user_id': user_id}

    myurl = "http://0.0.0.0:8180/predict"
    req = urllib.request.Request(myurl)
    req.add_header('Content-Type', 'application/json; charset=utf-8')
    jsondata = json.dumps(body)
    jsondataasbytes = jsondata.encode('utf-8')   # needs to be bytes
    req.add_header('Content-Length', len(jsondataasbytes))
    #print (jsondataasbytes)
    response = urllib.request.urlopen(req, jsondataasbytes)
    return json.loads(response.read())

@app.route("/")
def index():
    return render_template('index.html')


@app.route('/predicted/<response>')
def predicted(response):
    response = json.loads(response)
    return render_template('predicted.html', response=response)


@app.route('/predict_form', methods=['GET', 'POST'])
def predict_form():
    form = ClientDataForm()
    data = dict()
    if request.method == 'POST':
        data['title'] = request.form.get('title')
        data['user_id'] = request.form.get('user_id')

        try:
            response = json.dumps(get_prediction(data['title'], data['user_id']))
        except ConnectionError:
            response = json.dumps({"error": "ConnectionError"})
        return redirect(url_for('predicted', response=response))
    return render_template('form.html', form=form)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8181, debug=False)
