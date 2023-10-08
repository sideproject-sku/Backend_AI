from flask import Flask, render_template, Response
from time import sleep
from generate import main

app = Flask(__name__)


@app.route('/')
def Index():
    return render_template('index.html')  # index.html 파일을 렌더링하여 반환합니다.

@app.route('/stream')
def Stream():
    return Response(main(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(host="127.0.0.1", port="8080")
