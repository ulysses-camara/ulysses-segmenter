import os

import flask
import flask_cors


app = flask.Flask(__name__)
flask_cors.CORS(app)

data = list(map(lambda item: {"token": item[0], "label": item[1]}, zip(
    "Please use the Python API to send (and retrieve) "
    "data to this front-end ('python_api.py' module).".split(),
    9 * [0] + [1] + 3 * [0] + [2, 3],
)))

modified: list[bool] = []


@app.route("/refinery-data-transfer", methods=["GET", "POST"])
def data_transfer():
    if flask.request.method == "POST":
        global data
        data = flask.request.get_json()
        return ("Ok", 200)

    get_response = flask.jsonify(data)
    get_response.headers.add('Access-Control-Allow-Origin', '*')

    return get_response


@app.route("/", methods=["GET"])
def home():
    return flask.redirect(flask.url_for("static", filename="index.html"))
