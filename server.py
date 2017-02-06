#! /usr/bin/env python3

import os
from flask import Flask, render_template, jsonify, request
app = Flask(__name__, template_folder='frontend', static_folder='frontend')


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/recognize')
def compute():
    pixels = request.args['pixels']
    command = './mnist_in_mpi.out predict ' + pixels

    output = os.popen(command).read()  # very hackish, very bad, don't do this
    digit_predicted = int(output.split()[0])
    confidence = float(output.split()[1])

    return jsonify({'number': digit_predicted, 'confidence': confidence})

app.run()
