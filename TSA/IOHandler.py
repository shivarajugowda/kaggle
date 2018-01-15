import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation
from PIL import Image
import io
from lxml import etree
import pickle
import re
import cv2
import time
from object_detection.utils import label_map_util

# Inclues (most of) the IO related functions.

STAGE = 'stage2'
IMAGES_FOLDER = 'images'
INCORRECT_IMAGES_FOLDER = 'incorrect_images'
DATA_FOLDER = '../kaggleDHS/data'
FORMAT_APS = "aps"
FORMAT_A3DAPS = "a3daps"
TRAINING_SET = STAGE + '_labels.csv'
SOLUTION_SET = STAGE + '_solution.csv'
SUBMIT_SET = STAGE + '_sample_submission.csv'
TEST_SET = STAGE + '_testset.csv'
ANNOTATION_FOLDER = os.path.join(IMAGES_FOLDER, "annotations")
ZONE_DETECTOR_MODEL_FILE = "zoneDetector.pickle.dat"
INPUTS_JSON="inputs.json"

DETECTED_OBJECTS_PKL_FILE = 'objectsDetected_' + STAGE + '.pkl'
CUSTOM_PREDICTED_DATA  = 'zone_data_custom.csv'
LABELED_PREDICTED_DATA = 'zone_data_label.csv'


PATH_TO_CKPT =  'inference/frozen_inference_graph.pb'
PATH_TO_LABELS = 'train/label_map.pbtxt'
MAX_NUM_CLASSES = 10

JPG_WIDTH = 512
JPG_HEIGHT = 660

# Load label map
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=MAX_NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)


def getAnnotationFolder(imgFolder):
    return os.path.join(imgFolder, "annotations")

def _read_header(infile):
    """Read image header (first 512 bytes)
       Borrowed from William Cukierski's kernel(https://www.kaggle.com/wcukierski/reading-images)
    """
    h = dict()
    fid = open(infile, 'r+b')
    h['filename'] = b''.join(np.fromfile(fid, dtype = 'S1', count = 20))
    h['parent_filename'] = b''.join(np.fromfile(fid, dtype = 'S1', count = 20))
    h['comments1'] = b''.join(np.fromfile(fid, dtype = 'S1', count = 80))
    h['comments2'] = b''.join(np.fromfile(fid, dtype = 'S1', count = 80))
    h['energy_type'] = np.fromfile(fid, dtype = np.int16, count = 1)
    h['config_type'] = np.fromfile(fid, dtype = np.int16, count = 1)
    h['file_type'] = np.fromfile(fid, dtype = np.int16, count = 1)
    h['trans_type'] = np.fromfile(fid, dtype = np.int16, count = 1)
    h['scan_type'] = np.fromfile(fid, dtype = np.int16, count = 1)
    h['data_type'] = np.fromfile(fid, dtype = np.int16, count = 1)
    h['date_modified'] = b''.join(np.fromfile(fid, dtype = 'S1', count = 16))
    h['frequency'] = np.fromfile(fid, dtype = np.float32, count = 1)
    h['mat_velocity'] = np.fromfile(fid, dtype = np.float32, count = 1)
    h['num_pts'] = np.fromfile(fid, dtype = np.int32, count = 1)
    h['num_polarization_channels'] = np.fromfile(fid, dtype = np.int16, count = 1)
    h['spare00'] = np.fromfile(fid, dtype = np.int16, count = 1)
    h['adc_min_voltage'] = np.fromfile(fid, dtype = np.float32, count = 1)
    h['adc_max_voltage'] = np.fromfile(fid, dtype = np.float32, count = 1)
    h['band_width'] = np.fromfile(fid, dtype = np.float32, count = 1)
    h['spare01'] = np.fromfile(fid, dtype = np.int16, count = 5)
    h['polarization_type'] = np.fromfile(fid, dtype = np.int16, count = 4)
    h['record_header_size'] = np.fromfile(fid, dtype = np.int16, count = 1)
    h['word_type'] = np.fromfile(fid, dtype = np.int16, count = 1)
    h['word_precision'] = np.fromfile(fid, dtype = np.int16, count = 1)
    h['min_data_value'] = np.fromfile(fid, dtype = np.float32, count = 1)
    h['max_data_value'] = np.fromfile(fid, dtype = np.float32, count = 1)
    h['avg_data_value'] = np.fromfile(fid, dtype = np.float32, count = 1)
    h['data_scale_factor'] = np.fromfile(fid, dtype = np.float32, count = 1)
    h['data_units'] = np.fromfile(fid, dtype = np.int16, count = 1)
    h['surf_removal'] = np.fromfile(fid, dtype = np.uint16, count = 1)
    h['edge_weighting'] = np.fromfile(fid, dtype = np.uint16, count = 1)
    h['x_units'] = np.fromfile(fid, dtype = np.uint16, count = 1)
    h['y_units'] = np.fromfile(fid, dtype = np.uint16, count = 1)
    h['z_units'] = np.fromfile(fid, dtype = np.uint16, count = 1)
    h['t_units'] = np.fromfile(fid, dtype = np.uint16, count = 1)
    h['spare02'] = np.fromfile(fid, dtype = np.int16, count = 1)
    h['x_return_speed'] = np.fromfile(fid, dtype = np.float32, count = 1)
    h['y_return_speed'] = np.fromfile(fid, dtype = np.float32, count = 1)
    h['z_return_speed'] = np.fromfile(fid, dtype = np.float32, count = 1)
    h['scan_orientation'] = np.fromfile(fid, dtype = np.int16, count = 1)
    h['scan_direction'] = np.fromfile(fid, dtype = np.int16, count = 1)
    h['data_storage_order'] = np.fromfile(fid, dtype = np.int16, count = 1)
    h['scanner_type'] = np.fromfile(fid, dtype = np.int16, count = 1)
    h['x_inc'] = np.fromfile(fid, dtype = np.float32, count = 1)
    h['y_inc'] = np.fromfile(fid, dtype = np.float32, count = 1)
    h['z_inc'] = np.fromfile(fid, dtype = np.float32, count = 1)
    h['t_inc'] = np.fromfile(fid, dtype = np.float32, count = 1)
    h['num_x_pts'] = np.fromfile(fid, dtype = np.int32, count = 1)
    h['num_y_pts'] = np.fromfile(fid, dtype = np.int32, count = 1)
    h['num_z_pts'] = np.fromfile(fid, dtype = np.int32, count = 1)
    h['num_t_pts'] = np.fromfile(fid, dtype = np.int32, count = 1)
    h['x_speed'] = np.fromfile(fid, dtype = np.float32, count = 1)
    h['y_speed'] = np.fromfile(fid, dtype = np.float32, count = 1)
    h['z_speed'] = np.fromfile(fid, dtype = np.float32, count = 1)
    h['x_acc'] = np.fromfile(fid, dtype = np.float32, count = 1)
    h['y_acc'] = np.fromfile(fid, dtype = np.float32, count = 1)
    h['z_acc'] = np.fromfile(fid, dtype = np.float32, count = 1)
    h['x_motor_res'] = np.fromfile(fid, dtype = np.float32, count = 1)
    h['y_motor_res'] = np.fromfile(fid, dtype = np.float32, count = 1)
    h['z_motor_res'] = np.fromfile(fid, dtype = np.float32, count = 1)
    h['x_encoder_res'] = np.fromfile(fid, dtype = np.float32, count = 1)
    h['y_encoder_res'] = np.fromfile(fid, dtype = np.float32, count = 1)
    h['z_encoder_res'] = np.fromfile(fid, dtype = np.float32, count = 1)
    h['date_processed'] = b''.join(np.fromfile(fid, dtype = 'S1', count = 8))
    h['time_processed'] = b''.join(np.fromfile(fid, dtype = 'S1', count = 8))
    h['depth_recon'] = np.fromfile(fid, dtype = np.float32, count = 1)
    h['x_max_travel'] = np.fromfile(fid, dtype = np.float32, count = 1)
    h['y_max_travel'] = np.fromfile(fid, dtype = np.float32, count = 1)
    h['elevation_offset_angle'] = np.fromfile(fid, dtype = np.float32, count = 1)
    h['roll_offset_angle'] = np.fromfile(fid, dtype = np.float32, count = 1)
    h['z_max_travel'] = np.fromfile(fid, dtype = np.float32, count = 1)
    h['azimuth_offset_angle'] = np.fromfile(fid, dtype = np.float32, count = 1)
    h['adc_type'] = np.fromfile(fid, dtype = np.int16, count = 1)
    h['spare06'] = np.fromfile(fid, dtype = np.int16, count = 1)
    h['scanner_radius'] = np.fromfile(fid, dtype = np.float32, count = 1)
    h['x_offset'] = np.fromfile(fid, dtype = np.float32, count = 1)
    h['y_offset'] = np.fromfile(fid, dtype = np.float32, count = 1)
    h['z_offset'] = np.fromfile(fid, dtype = np.float32, count = 1)
    h['t_delay'] = np.fromfile(fid, dtype = np.float32, count = 1)
    h['range_gate_start'] = np.fromfile(fid, dtype = np.float32, count = 1)
    h['range_gate_end'] = np.fromfile(fid, dtype = np.float32, count = 1)
    h['ahis_software_version'] = np.fromfile(fid, dtype = np.float32, count = 1)
    h['spare_end'] = np.fromfile(fid, dtype = np.float32, count = 10)
    return h


def read_data(subject, format):
    """Read any of the 4 types of image files, returns a numpy array of the image contents
       Borrowed from William Cukierski's kernel(https://www.kaggle.com/wcukierski/reading-images)
    """
    format = FORMAT_APS if format is None else format
    infile = os.path.join(DATA_FOLDER, STAGE, format, subject + "." + format )

    extension = os.path.splitext(infile)[1]
    h = _read_header(infile)
    nx = int(h['num_x_pts'])
    ny = int(h['num_y_pts'])
    nt = int(h['num_t_pts'])
    fid = open(infile, 'rb')
    fid.seek(512) #skip header
    if extension == '.aps' or extension == '.a3daps':
        if(h['word_type']==7): #float32
            data = np.fromfile(fid, dtype = np.float32, count = nx * ny * nt)
        elif(h['word_type']==4): #uint16
            data = np.fromfile(fid, dtype = np.uint16, count = nx * ny * nt)
        data = data * h['data_scale_factor'] #scaling factor
        data = data.reshape(nx, ny, nt, order='F').copy() #make N-d image
    elif extension == '.a3d':
        if(h['word_type']==7): #float32
            data = np.fromfile(fid, dtype = np.float32, count = nx * ny * nt)
        elif(h['word_type']==4): #uint16
            data = np.fromfile(fid, dtype = np.uint16, count = nx * ny * nt)
        data = data * h['data_scale_factor'] #scaling factor
        data = data.reshape(nx, nt, ny, order='F').copy() #make N-d image
    elif extension == '.ahi':
        data = np.fromfile(fid, dtype = np.float32, count = 2* nx * ny * nt)
        data = data.reshape(2, ny, nx, nt, order='F').copy()
        real = data[0,:,:,:].copy()
        imag = data[1,:,:,:].copy()
    fid.close()
    if extension != '.ahi':
        return data
    else:
        return real, imag


def plot_image(subject, format):
    """Plot images as animation
       Derived from William Cukierski's kernel(https://www.kaggle.com/wcukierski/reading-images)
    """
    data = read_data(subject, format)
    fig = plt.figure(figsize=(7.5, 7.5))
    ax = fig.add_subplot(111)

    result = []
    lastFrameIdx = data.shape[2]-1
    for x in range(0, lastFrameIdx):
        im = ax.imshow(np.flipud(data[:, :, x].transpose()), cmap='viridis')
        result.append(im)

    def animate(i):
        return [result[i]]

    return matplotlib.animation.FuncAnimation(fig, animate, frames=range(0, lastFrameIdx), interval=300, blit=True)

# Load training labels,
# returns [ Subject, NumContrabands, Zones]
def load_solutionSet() :
    csvFile = os.path.join(DATA_FOLDER, STAGE, SOLUTION_SET)
    df = pd.read_csv(csvFile)

    df['Zone'] = df['Id'].str.rpartition('_')[2].str.replace("Zone", "")
    df['Subject'] = df['Id'].str.rpartition('_')[0]

    df = df.groupby('Subject').agg({'Zone': [lambda x: "%s" % ' '.join(x[df.ix[x.index]['Probability'] > 0])],
                                 'Probability': ['sum']
                                 }).rename(columns={'sum': 'NumContrabands', '<lambda>': 'Zones'})
    df.columns = df.columns.droplevel(0)
    df = df.reset_index()
    return df

# Load training labels,
# returns [ Subject, NumContrabands, Zones]
def load_trainingLabels() :
    csvFile = os.path.join(DATA_FOLDER, STAGE, TRAINING_SET)
    df = pd.read_csv(csvFile)

    df['Zone'] = df['Id'].str.rpartition('_')[2].str.replace("Zone", "")
    df['Subject'] = df['Id'].str.rpartition('_')[0]

    df = df.groupby('Subject').agg({'Zone': [lambda x: "%s" % ' '.join(x[df.ix[x.index]['Probability'] > 0])],
                                 'Probability': ['sum']
                                 }).rename(columns={'sum': 'NumContrabands', '<lambda>': 'Zones'})
    df.columns = df.columns.droplevel(0)
    df = df.reset_index()
    #dfTest = load_testSet()
    #df = df[~df['Subject'].isin(dfTest['Subject'])]
    return df

# Load testing set
# returns [ Subject ]
def load_submitSet() :
    csvFile = os.path.join(DATA_FOLDER, STAGE, SUBMIT_SET)
    df = pd.read_csv(csvFile)

    df['Subject'] = df['Id'].str.rpartition('_')[0]
    return df[['Subject']].drop_duplicates()

# Load testing set
# returns [ Subject ]
def load_testSet() :
    csvFile = os.path.join(DATA_FOLDER, STAGE, TEST_SET)
    df = pd.read_csv(csvFile, keep_default_na=False)
    return df

# Append zero to single digit numbers, helps out when listing files.
def getFileNameFromSubjAndFrame(subj, frame) :
    num = str(frame) if frame >= 10 else '0' + str(frame)
    return subj + "_" + num

# Get Subject and Frame from FileName
def getSubjAndFrameFromFileName(filename) :
    strs = filename.split("_")
    subj = strs[0]
    frame = int(strs[1].split(".")[0])
    return subj, frame

# Return true if subject is in labeled set.
def isInLabeledSet(subj) :
    file = None
    for filename in os.listdir(IMAGES_FOLDER):
        if filename.startswith(subj) :
            file = filename
            break;
    return file is not None

'''
    Returns Image data as an numpy array for all frames of a subject 
    [numposes, Height, Width, 3]
'''
def getImageData(sujbect, format):
    data = read_data(sujbect, format)
    result = []
    for x in range(0, data.shape[2]):
        img_np = np.flipud(data[:, :, x].transpose())
        img_np *= 1.0 / img_np.max()  # Normalize image
        img_np = np.uint8(plt.cm.viridis(img_np) * 255)  # Apply colormap
        result.append(img_np[:,:,:3])

    # The end dimension is [numposes, Height, Width, 3]
    return np.asarray(result)

def save_frame(imgFolder, subject, frame, format):
    data = read_data(subject, format)
    outName = getFileNameFromSubjAndFrame(subject, frame) + ".jpg"
    outPath = os.path.join(imgFolder,  outName)

    img_np = np.flipud(data[:,:,frame].transpose())
    img_np *= 1.0 / img_np.max()  # Normalize image
    img_np = np.uint8(plt.cm.viridis(img_np) * 255) # Apply colormap

    image = Image.fromarray(img_np[:,:,:3])
    image.save(outPath, "JPEG")

    return

def save_images(imgFolder, subject, format):
    data = read_data(subject, format)
    for x in range(0, data.shape[2]) :
        save_frame(imgFolder, subject, x, format)
    return



# Add bounding box
def _addBox(annotation, label, box) :
    if box is None :
        return
    object = etree.SubElement(annotation, "object")
    name = etree.SubElement(object, "name")
    name.text = label
    bndbox = etree.SubElement(object, "bndbox")
    xmin = etree.SubElement(bndbox, "xmin")
    xmin.text = str(int(box[1] * JPG_WIDTH))
    ymin = etree.SubElement(bndbox, "ymin")
    ymin.text = str(int(box[0] * JPG_HEIGHT))
    xmax = etree.SubElement(bndbox, "xmax")
    xmax.text = str(int(box[3] * JPG_WIDTH))
    ymax = etree.SubElement(bndbox, "ymax")
    ymax.text = str(int(box[2] * JPG_HEIGHT))

def generateAnnotation(imgFolder, subj, frame, head, hands, groin, contrabands) :
    fileBaseName = getFileNameFromSubjAndFrame(subj, frame)
    annotation = etree.Element("annotation")
    folder = etree.SubElement(annotation, "folder")
    folder.text = imgFolder
    file = etree.SubElement(annotation, "filename")
    file.text = fileBaseName + ".jpg"
    size = etree.SubElement(annotation, "size")
    width = etree.SubElement(size, "width")
    width.text = str(JPG_WIDTH)
    height = etree.SubElement(size, "height")
    height.text = str(JPG_HEIGHT)
    segmented = etree.SubElement(annotation, "segmented")
    segmented.text = str(0)

    _addBox(annotation, "Head",head)
    _addBox(annotation, "Groin", groin)

    for hd in hands:
        _addBox(annotation, "Hand", hd)

    for cb in contrabands:
        _addBox(annotation, "Contraband", cb[1])

    et = etree.ElementTree(annotation)
    et.write(os.path.join(getAnnotationFolder(imgFolder), fileBaseName + ".xml"), pretty_print=True)


def writeResults(df) :
    df.to_csv('out.csv')

def getZoneDetectorModel() :
    # load model from file
    model = pickle.load(open(ZONE_DETECTOR_MODEL_FILE, "rb"))
    return  model

def saveZoneDetectorModel(model) :
    # Save the Model
    pickle.dump(model, open(ZONE_DETECTOR_MODEL_FILE, "wb"))

