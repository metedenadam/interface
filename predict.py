from load_graph import *
from darkflow.cython_utils.cy_yolo2_findboxes import box_constructor
import json, math, os, cv2, argparse, numpy as np, tensorflow as tf
from multiprocessing.pool import ThreadPool

pool = ThreadPool()

def process_box(meta, b, h, w, threshold):
	max_indx = np.argmax(b.probs)
	max_prob = b.probs[max_indx]
	label = meta['labels'][max_indx]
	if max_prob > threshold:
		left  = int ((b.x - b.w/2.) * w)
		right = int ((b.x + b.w/2.) * w)
		top   = int ((b.y - b.h/2.) * h)
		bot   = int ((b.y + b.h/2.) * h)
		if left  < 0    :  left = 0
		if right > w - 1: right = w - 1
		if top   < 0    :   top = 0
		if bot   > h - 1:   bot = h - 1
		mess = '{}'.format(label)
		return (left, right, top, bot, mess, max_indx, max_prob)
	return None

def postprocess(net_out, im, meta):
	"""
	Takes net output, draw predictions, save to disk
	"""
	outfolder = os.path.join(inp_path, 'out')
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

		
		
	img_name = os.path.join(outfolder, os.path.basename(im))
	cv2.imwrite(img_name, imgcv)

def resize_input(im):
	h, w, c = meta['inp_size']
	imsz = cv2.resize(im, (w, h))
	imsz = imsz / 255.
	imsz = imsz[:,:,::-1]
	return imsz

def preprocess(im):
	if type(im) is not np.ndarray:
		im = cv2.imread(im)
	im = resize_input(im)
	return im

def is_inp(name): 
    return name.lower().endswith(('.jpg', '.jpeg', '.png'))

def findboxes(meta, net_out):
    boxes = list()
    boxes=box_constructor(meta,net_out)
    return boxes

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--frozen_model_filename", default="built_graph/yolov2_logos.pb", type=str, help="Frozen model file to import")
    args = parser.parse_args()

    with open("./built_graph/yolov2_logos.meta","r") as f:
        meta_file = f.read()
    
    meta = json.loads(meta_file)

    graph = load_graph(args.frozen_model_filename)
    x = graph.get_tensor_by_name('input:0')
    y = graph.get_tensor_by_name('output:0')
    inp_path = "./images/"
    all_inps = os.listdir(inp_path)
    all_inps = [i for i in all_inps if is_inp(i)]
    if not all_inps:
        msg = 'Failed to find any images in {} .'
        exit('Error: {}'.format(msg.format(inp_path)))

    batch = min(16, len(all_inps))
    n_batch = int(math.ceil(len(all_inps) / batch))
    for j in range(n_batch):
        from_idx = j * batch
        to_idx = min(from_idx + batch, len(all_inps))

        this_batch = all_inps[from_idx:to_idx]
        inp_feed = pool.map(lambda inp: (
            np.expand_dims(preprocess(
                os.path.join(inp_path, inp)), 0)), this_batch)

        
    # We launch a Session
    with tf.Session(graph=graph) as sess:
        feed_dict = {x : np.concatenate(inp_feed, 0)}    
        y_out = sess.run(y, feed_dict)
        pool.map(lambda p: (lambda i, prediction:
            postprocess(
               prediction, os.path.join(inp_path, this_batch[i]), meta))(*p),
            enumerate(y_out))

        print("Yey")