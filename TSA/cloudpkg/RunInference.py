import tensorflow as tf
import pickle
import time
import numpy as np
import json
import base64
import sys

#INPUTS_JSON = "inputs.json"
#PATH_TO_CKPT = 'inference_SSD/cloudPrediction/frozen_inference_graph.pb'
#PICKLE_FILE = 'objectsDetected_aps.pkl'

tf.logging.set_verbosity(tf.logging.INFO)

flags = tf.app.flags
flags.DEFINE_string('graph_file_path', '',
                    'Path to the inference graph file.')
flags.DEFINE_string('input_path', '',
                    'Path to the input JSON file.')
flags.DEFINE_string('output_path', '',
                    'Path to the output pickle file.')
FLAGS = flags.FLAGS

def getTFSession(graphfile) :
    # Load a frozen TensorFlow model into memory.
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(graphfile, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')

            sess = tf.Session(graph=detection_graph)

            return sess, detection_graph

def detectObjects(sess, detection_graph, frames):
    # Definite input and output Tensors for detection_graph
    input_tensor = detection_graph.get_tensor_by_name('encoded_image_string_tensor:0')
    # Each box represents a part of the image where a particular object was detected.
    detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
    # Each score represent how level of confidence for each of the objects.
    # Score is shown on the result image, together with the class label.
    detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
    detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
    num_detections = detection_graph.get_tensor_by_name('num_detections:0')

    npBoxes, npScores, npClasses = ([] for i in range(3))
    for frame in frames :
        imgbytes = base64.decodestring(frame)
        # Actual detection.
        (boxes, scores, classes, num) = sess.run(
            [detection_boxes, detection_scores, detection_classes, num_detections],
            feed_dict={input_tensor:[imgbytes]})

        boxes = np.squeeze(boxes)
        scores = np.squeeze(scores)
        classes = np.squeeze(classes)

        npBoxes.append(boxes)
        npScores.append(scores)
        npClasses.append(classes)

    return np.asarray(npBoxes), np.asarray(npScores), np.asarray(npClasses)


def main():
    assert FLAGS.input_path, '`input_path` is missing.'
    assert FLAGS.output_path, '`output_path` is missing.'
    assert FLAGS.graph_file_path, '`graph_file_path` is missing.'

    inputfile = FLAGS.input_path
    outputfile = FLAGS.output_path
    graphfile = FLAGS.graph_file_path

    print 'Input file is :', inputfile
    print 'Output file is :', outputfile
    print 'Inference graph file is :', graphfile

    starttime = time.time()

    objects = {}
    if tf.gfile.Exists(outputfile):
        with tf.gfile.GFile(outputfile, 'rb') as handle:
            objects = pickle.load(handle)

    time1 = time.time()
    sess, detection_graph = getTFSession(graphfile)
    print 'Time to initialize TF Session = %0.3f ms' % ((time.time() - time1) * 1000.0)

    with tf.gfile.GFile(inputfile, 'r') as jsonFile:
        data = json.load(jsonFile)

    count = 0;
    for subj, frames in data.iteritems():
        count += 1
        print 'Count = ', count,

        time1 = time.time()
        boxes, scores, classes = detectObjects(sess, detection_graph, frames)
        print ' Detecting Labels for %s took %0.3f ms' % (subj, (time.time() - time1) * 1000.0)

        objects[subj] = {'boxes': boxes, 'scores': scores, 'classes': classes}
        with tf.gfile.GFile(outputfile, 'wb') as handle:
            pickle.dump(objects, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print 'Total time taken = %0.3f ms' % ((time.time() - starttime) * 1000.0)

if __name__ == "__main__":
    main()
