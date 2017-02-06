#! /usr/bin/env python3

from flask import Flask, render_template, jsonify, request
app = Flask(__name__, template_folder='frontend', static_folder='frontend')


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/recognize')
def compute():
    pixels = request.args['pixels']
    return jsonify({'number': 9, 'confidence': .98})  # TODO: get from c++


app.run()
