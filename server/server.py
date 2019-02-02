import sys
sys.path.append('..')

from flask import Flask, request, render_template, redirect, url_for, send_from_directory
from werkzeug.utils import secure_filename
from utils import utils
import os
import torch
from models.baseline import ConvBlock, UpConvBlock, Baseline
from flask import send_from_directory

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
device = "cpu"

def load_net(device=device):
    net = Baseline(device,
              ConvBlock(3, 64, 7),
    #           ConvBlock(32, 64, 7),
    #           ConvBlock(64, 128, 7),
              UpConvBlock(64, upscale=2),
              ConvBlock(64, 3)
            )

    net.load_state_dict(torch.load("state_dict.wght"))
    net.eval()
    return net

@app.route("/", methods=['GET', 'POST'])
def hello():
    if request.method == 'POST':
        if 'photo' not in request.files:
            return render_template('error.html', message='No file part')

        file = request.files['photo']
        if file.filename == '':
            return render_template('error.html', message='No filename')

        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        transformed_filename, pred_filename = utils.process_image(filepath, load_net())

        return render_template('prediction.html', initial=transformed_filename, pred=pred_filename)
    return render_template('index.html')



@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


if __name__ == '__main__':
      app.run(host='0.0.0.0', port=1337)
