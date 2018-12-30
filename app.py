import json, math, os, cv2, argparse, numpy as np, tensorflow as tf
from darkflow.cython_utils.cy_yolo2_findboxes import box_constructor
from multiprocessing.pool import ThreadPool
from flask import Flask, render_template, request, redirect, url_for
from predict import load_graph, preprocess, process_box, findboxes


app = Flask(__name__)
app.secret_key = '+_9o$w9+9xro!-y(wvuv+vvyc!$x(@ak(!oh@ih0ul+6cf=$f'
app.config['UPLOAD_FOLDER'] = "./static/"
ALLOWED_EXTENSIONS = set(['jpg'])

def postprocess(net_out, im, meta):
    """
    Takes net output, draw predictions, save to disk
    """
    threshold = -0.1
    colors, labels = meta['colors'], meta['labels']

    boxes = findboxes(meta, net_out)

    if type(im) is not np.ndarray:
        imgcv = cv2.imread(im)
    else: imgcv = im

    h, w, _ = imgcv.shape
    for b in boxes:
        boxResults = process_box(meta, b, h, w, threshold)
        if boxResults is None:
            continue
        left, right, top, bot, mess, max_indx, confidence = boxResults
        thick = int((h + w) // 300)

        cv2.rectangle(imgcv,
            (left, top), (right, bot),
            meta['colors'][max_indx], thick)
        cv2.putText(
            imgcv, mess, (left, top - 12),
            0, 1e-3 * h, meta['colors'][max_indx],
            thick // 3)

    # img_name = os.path.join(outfolder, os.path.basename(im))
    img_name = os.path.join(app.config['UPLOAD_FOLDER'], "img.jpg")
    cv2.imwrite(img_name, imgcv)
    # return imgcv

def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS

@app.route('/', methods = ['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        try:
            file = request.files['file']
        except:
            errormsg = "Please select a file!"
            return render_template('upload.html', errormsg = errormsg)
        try:
            os.remove("./static/img.jpg")
        except:
            pass
        if file and allowed_file(file.filename):
            imgtostr = np.fromstring(file.read(), np.uint8)
            img = cv2.imdecode(imgtostr, cv2.IMREAD_UNCHANGED)
            inp_feed = np.expand_dims(preprocess(img, meta), 0)
            feed_dict = feed_dict = {x : inp_feed}
            y_out = sess.run(y, feed_dict)
            ThreadPool().map(lambda p: (lambda i, prediction:
                postprocess(
                prediction, img, meta))(*p),
                enumerate(y_out))
            return redirect(url_for('upload_file'))
        else:
            errormsg = "File type not supported"
            return render_template('upload.html', errormsg = errormsg)
    else:
        errormsg = "Please upload a file:"
        return render_template('upload.html', errormsg = errormsg) 

if __name__ == "__main__":
    try:
        os.remove("./static/img.jpg")
    except:
        pass
    with open("./built_graph/yolov2_logos.meta","r") as f:
        meta_file = f.read()
    graph = load_graph("built_graph/yolov2_logos.pb")
    meta = json.loads(meta_file)
    x = graph.get_tensor_by_name('input:0')
    y = graph.get_tensor_by_name('output:0')
    sess = tf.Session(graph=graph)
    app.run(host="0.0.0.0", port=5555)

