import os
from flask import Flask, render_template, request, url_for, Response
import requests
from flask_cors import CORS, cross_origin
from werkzeug.utils import secure_filename
from app.utils import get_all_speakers, preprocess_all, preprocess_test, remove_test_file
from app.CNN import trainCNN, testCNN, get_speaker_name, reset_speaker_name
import json 

app = Flask(__name__)
CORS(app)
app.config.from_pyfile('config.py')


@app.route('/')
def main():
    return render_template('index.html')


@app.route('/enrollSpeaker')
def enrollSpeaker():
    speakers = get_all_speakers()
    # random_para = []
    # for i in range(1):
    #     r = requests.get("http://pastehtml.com/view/1c0gckz.html")
    #     print(r.json())
    return render_template('enroll_speaker.html', speakers=speakers)


@app.route('/get_speaker_name')
def return_name():
    speaker_name = get_speaker_name()
    
    # while speaker_name == "None":
        # time.sleep(1)
        # speaker_name = get_speaker_name()

    reset_speaker_name()
    return json.dumps({'name': speaker_name}), 200, {'ContentType':'application/json'} 


@app.route('/trainModel', methods = ['POST'])
def trainModel():
    if request.method == 'POST':
        preprocess_all()
        trainCNN()
    return render_template('index.html')
	# https://www.youtube.com/watch?v=f6Bf3gl4hWY
	# https://blog.keras.io/building-a-simple-keras-deep-learning-rest-api.html


@app.route('/recognizeSpeaker')
def recognizeSpeaker():
	return render_template('recognize_speaker.html')


@app.route('/upload', methods = ['POST'])
@cross_origin()
def upload():
    if request.method == 'POST':
        file_type = request.form['file_type']
        file = request.files['file']
        if file_type == 'train':
            file.save(os.path.join(app.config['RAW_TRAIN_FOLDER'], file.filename))
        else:
            file.save(os.path.join(app.config['RAW_TEST_FOLDER'], file.filename))
            preprocess_test(file.filename)
            testCNN()
            remove_test_file()
    return 'ok'


@app.route('/register')
def register():
	pass


if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.debug = True
    app.run(host='0.0.0.0', port=port)