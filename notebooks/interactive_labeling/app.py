"""Communication between Jupyter Notebook and front-end label refinery."""
import typing as t

import flask
import flask_cors
import werkzeug.wrappers


app = flask.Flask(__name__)
flask_cors.CORS(app)


app_state: dict[str, t.Any] = dict(
    data=list(
        map(
            lambda item: {"token": item[0], "label": item[1]},
            zip(
                "Please use the Python API to send (and retrieve) "
                "data to this front-end ('python_api.py' module).".split(),
                9 * [0] + [1] + 3 * [0] + [2, 3],
            ),
        )
    ),
    modified=[],
    need_refresh=False,
)


@app.route("/refinery-data-transfer", methods=["GET", "POST"])
def data_transfer() -> flask.Response:
    """Transfer data to and from Jupyter Notebook and front-end interface."""
    if flask.request.method == "POST":
        app_state["data"] = flask.request.get_json()
        return flask.make_response(dict(response="OK", status=200))

    get_response = flask.jsonify(app_state["data"])
    get_response.headers.add("Access-Control-Allow-Origin", "*")

    return get_response


@app.route("/call-for-refresh", methods=["GET", "POST"])
def call_for_refresh() -> flask.Response:
    """Control flag for whether front-end should auto-refresh the page."""
    if flask.request.method == "POST":
        app_state["need_refresh"] = True
        return flask.make_response(dict(response="OK", status=200))

    get_response = flask.jsonify({"need_refresh": app_state["need_refresh"]})
    get_response.headers.add("Access-Control-Allow-Origin", "*")
    app_state["need_refresh"] = False

    return get_response


@app.route("/", methods=["GET"])
def home() -> werkzeug.wrappers.Response:
    """Render label refinement front-end."""
    return flask.redirect(flask.url_for("static", filename="index.html"))
