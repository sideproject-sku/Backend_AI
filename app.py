from flask import Flask, render_template, Response, jsonify, request
from time import sleep
from generate import main

app = Flask(__name__)


@app.route('/')
def Index():
    return render_template('index.html')

@app.route('/stream/<cam_ID>')
def Stream(cam_ID):
    camera_ID = cam_ID
    returns = main(camera_ID)

    return Response(returns, mimetype='multipart/x-mixed-replace; boundary=frame')



#(b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + res + b'\r\n')


@app.route('/api/<camera_ID>') #rest GET
def get_echo_call(camera_ID):
    lnk = "127.0.0.1/stream/"+ camera_ID
    return jsonify({"cameraId" : camera_ID ,"url": lnk})




@app.route('/api', methods=['POST']) #rest POST
def post_echo_call():
    param = request.get_json()
    print(param)
    return jsonify(param)



if __name__ == "__main__":
    app.run(host="127.0.0.1", port="8080")
