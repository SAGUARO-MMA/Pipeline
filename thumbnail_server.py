#!/usr/bin/env python

from flask import Flask, send_from_directory
import settings

app = Flask(__name__)


@app.route('/api/<path:filename>')
def download_file(filename):
    return send_from_directory(settings.THUMB_PATH + '/', filename, as_attachment=True)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5013, debug=True)
