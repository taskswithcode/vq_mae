from flask import Flask,request
import gen_transform as gen
import os
import sys

import logging
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

app = Flask(__name__)

model = None
device = None
counter = 1

@app.route('/search', methods=['GET'])
def hello():
    global device,model,counter
    os.environ['WERKZEUG_RUN_MAIN'] = 'true'
    if (model == None):
        device,model = gen.load_model()
    args = request.args
    name = args.get('name')
    output_file = "generated/" + ".".join(name.split("/")[-1].split(".")[:-1]) + ".npy"
    if (counter % 10 == 0):
        print("{:,}".format(counter), "]", "file name in flask",output_file,file=sys.stderr)
    counter += 1
    gen.loc_convert(device,model,name,output_file)
    return output_file
