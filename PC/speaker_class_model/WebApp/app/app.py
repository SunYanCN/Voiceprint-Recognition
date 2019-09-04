import os
from flask import Flask, render_template, request
from flask_cors import CORS, cross_origin
from app.utils import get_all_speakers, preprocess_all, preprocess_test, remove_test_file
import json
from kws import train_cnn, test_cnn, get_speaker_name, reset_speaker_name

app = Flask(__name__)
CORS(app)
app.config.from_pyfile('config.py')


@app.route('/')
def main():
    return render_template('index.html')


@app.route('/enrollSpeaker')
def enrollSpeaker():
    speakers = get_all_speakers()
    print(speakers)
    return render_template('enroll_speaker.html', speakers=speakers)


@app.route('/get_speaker_name')
def return_name():
    speaker_name = get_speaker_name()
    reset_speaker_name()
    return json.dumps({'name': speaker_name}), 200, {'ContentType':'application/json'} 


@app.route('/trainModel', methods = ['POST'])
def trainModel():
    if request.method == 'POST':
        preprocess_all()
        train_cnn(get_all_speakers())
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
        dir = file.filename.split("_")[0]
        if file_type == 'train':
            temp_path = os.path.join(app.config['RAW_TRAIN_FOLDER'], dir)
            if not os.path.exists(temp_path):
                os.mkdir(temp_path)
            train_save_path = os.path.join(temp_path, file.filename)
            file.save(train_save_path)
        else:
            temp_path = os.path.join(app.config['RAW_TEST_FOLDER'], dir)
            if not os.path.exists(temp_path):
                os.mkdir(temp_path)
            test_save_path = os.path.join(temp_path, file.filename)
            file.save(test_save_path)
            preprocess_test()
            test_cnn()
            remove_test_file()
    return 'ok'


@app.route('/register')
def register():
	pass


if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.debug = True
    app.run(host='0.0.0.0', port=port, threaded=False)