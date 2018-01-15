import os
import IOHandler
import pandas as pd
import numpy as np
from lxml import etree
import pickle
import random

MIN_SCORE_THRESHOLD = 0.5
def getObjectsInFrame(boxes, scores, classes) :
    head = None
    groin = None
    headMaxScore = 0
    groinMaxScore = 0
    contrabands = []
    hands = []
    for i in range(boxes.shape[0]):
        if scores[i] > MIN_SCORE_THRESHOLD:
            box = tuple(boxes[i,:].tolist())
            class_name = IOHandler.category_index[classes[i]]['name']
            if class_name == 'Head':
                if scores[i] > 0.8 and scores[i] > headMaxScore :
                    headMaxScore = scores[i]
                    head = box
            elif class_name == 'Hand':
                hands.append(box)
            elif class_name == 'Groin':
                if scores[i] > 0.5 and scores[i] > groinMaxScore :
                    groinMaxScore = scores[i]
                    groin = box
            elif class_name == 'Contraband':
                contrabands.append((scores[i],box))
    return head, hands, groin, contrabands

# Returns the average place of head/groin for all the frames.
def getAvgHeadAndGroin(boxes, scores, classes) :
    avgHead, avgGroin, avgHand = np.zeros(4), np.zeros(4), np.zeros(2)
    numHeads, numGroin, numHands = 0, 0, 0
    for frame in range(boxes.shape[0]):
        head, hands, groin, contrabands = getObjectsInFrame(boxes[frame, :, :], scores[frame, :], classes[frame, :])
        if head is not None :
            avgHead += head
            numHeads += 1
        if groin is not None :
            avgGroin += groin
            numGroin += 1
        for hand in hands:
            avgHand[0] = hand[0]
            avgHand[1] = hand[2]
            numHands += 1

    avgHead = avgHead/numHeads
    avgGroin = avgGroin/numGroin
    avgHand = avgHand/ numGroin

    return avgHead, avgGroin, avgHand

def getLabelsFromAnnotation(subj, frame, zonesTrue) :
    filename = IOHandler.getFileNameFromSubjAndFrame(subj,frame) + ".xml"

    xml = etree.parse(os.path.join(IOHandler.ANNOTATION_FOLDER, filename))
    width = float(xml.xpath("//annotation/size/width/text()")[0])
    height = float(xml.xpath("//annotation/size/height/text()")[0])

    head = None
    groin = None
    hands = []
    contrabands = {}

    for elem in xml.iter("object"):
        name = elem.xpath("./name/text()")[0]

        xmin = float(elem.xpath("./bndbox/xmin/text()")[0])/width
        ymin = float(elem.xpath("./bndbox/ymin/text()")[0])/height
        xmax = float(elem.xpath("./bndbox/xmax/text()")[0])/width
        ymax = float(elem.xpath("./bndbox/ymax/text()")[0])/height

        box = (ymin, xmin, ymax, xmax)

        if name == 'Head' :
            if head is not None:
                print "Multiple Heads found in ", filename
            head = box
        elif name == 'Hand':
            hands.append(box)
            if len(hands) > 2 :
                print "More than two hands found in ", filename
        elif name == 'Groin':
            if groin is not None:
                print "Multiple Groins found in ", filename
            groin = box
        elif name.startswith('Zone_'):
            zone = int(name.replace('Zone_', ''))
            if zone not in zonesTrue :
                print "Incorrect zone ", zone, " found. expecting one of ", zonesTrue," file = ", filename
            if zone in contrabands :
                print "Duplicate zones in one frame, Zone = ", zone, " file = ", filename
            contrabands[zone] = box
            if frame in [10] and zone in [14] :
                if box[1] > 0.5 and box[3] > 0.5 :
                    print "Zone = ", zone, "in Frame ", frame, " looks incorrect file = ", filename
            if groin and zone in [1,2,3,4,5,17] and box[0] > groin[0] :
                print "Zone = ", zone, "in Frame ", frame, " looks incorrect upper body below groin, file = ", filename
            if groin and zone in [11,12,13,14,15,16] and box[2] < groin[0] :
                print "Zone = ", zone, "in Frame ", frame, " looks incorrect lower body above groin, file = ", filename
            if frame in [1, 0, 15]:
                if head and zone == 2 and box[3] > head[3] :
                    print "Zone = ", zone, "in Frame ", frame, " looks incorrect, file = ", filename

        else :
            print "Unknown object name, ", name, " found in ", filename
            continue

    if head is None :
        print "Head not found in ", filename

    # Groin always below Head.
    if groin is not None and (groin[0] < head[0] or groin[2] < head[2]):
        print "Groin is above head in ", filename

    # Hands above Head.
    for hand in hands:
        if not (hand[2] < head[2]):
            print "Hand below head in ", filename

    return head, groin, hands, contrabands

def randomInnerRectangle(outerbox):
    yalpha = (outerbox[2] - outerbox[1]) * random.uniform(0.1, 0.5)
    xalpha = (outerbox[3] - outerbox[0]) * random.uniform(0.1, 0.5)
    return (outerbox[0] + yalpha, outerbox[1] + xalpha,
            outerbox[2] - yalpha, outerbox[3] - xalpha)
def getLabels(annotationFiles) :
    pkl_file = open(IOHandler.DETECTED_OBJECTS_PKL_FILE, 'rb')
    objdict = pickle.load(pkl_file)
    pkl_file.close()

    df = pd.concat([IOHandler.load_trainingLabels(), IOHandler.load_testSet()])
    zonedata = pd.DataFrame(columns=['Frame',
                                     'Head_0', 'Head_1', 'Head_2', 'Head_3',
                                     'Groin_1', 'Groin_2', 'Groin_3', 'Groin_4',
                                     #'Groin_1', 'Groin_3',
                                     'Hand_0', 'Hand_1',
                                     'CB_1', 'CB_2', 'CB_3', 'CB_4',
                                     'Zone'
                                     ])

    for filename in annotationFiles :
        strs = filename.split("_")
        subj = strs[0]
        frame = int(strs[1].split(".")[0])

        zoneStr = df[df['Subject'] == subj]['Zones'].iloc[0]
        zonesTrue = [] if not zoneStr else [int(k) for k in zoneStr.split(' ')]

        list = objdict[subj]
        boxes = list['boxes']
        scores = list['scores']
        classes = list['classes']

        avgHead, avgGroin, avgHand = getAvgHeadAndGroin(boxes, scores, classes)
        head, groin, hands, cbs = getLabelsFromAnnotation(subj, frame, zonesTrue)
        if groin is None :
            groin = avgGroin

        for zone, box in cbs.iteritems():
            input = [float(frame)/boxes.shape[0]]
            input.extend(head)
            input.extend(groin)
            #input.extend((groin[0], groin[2]))
            input.extend(avgHand)
            input.extend(box)

            input.append(zone)
            zonedata.loc[len(zonedata)] = input
            # Data Augmentation.
            if(zone in [0,8,1,15,7,9]) :
                for i in range(2) :
                    randBox = randomInnerRectangle(box);
                    input = [float(frame) / boxes.shape[0]]
                    input.extend(head)
                    input.extend(groin)
                    # input.extend((groin[0], groin[2]))
                    input.extend(avgHand)
                    input.extend(randBox)
                    input.append(zone)
                    zonedata.loc[len(zonedata)] = input

    return zonedata

def predictZone(model, X_test) :
    x = np.array(X_test)
    x = np.expand_dims(x, axis=0)
    # make predictions for test data
    y_pred = model.predict(x)
    predictions = [int(value) for value in y_pred]
    return predictions[0]

def testLabels(annotationFiles) :
    pkl_file = open(IOHandler.DETECTED_OBJECTS_PKL_FILE, 'rb')
    objdict = pickle.load(pkl_file)
    pkl_file.close()

    df = pd.concat([IOHandler.load_trainingLabels(), IOHandler.load_testSet()])
    zoneModel = IOHandler.getZoneDetectorModel()

    count = 0
    incorrectCount = 0
    for filename in annotationFiles :
        strs = filename.split("_")
        subj = strs[0]
        frame = int(strs[1].split(".")[0])

        zoneStr = df[df['Subject'] == subj]['Zones'].iloc[0]
        zonesTrue = [] if not zoneStr else [int(k) for k in zoneStr.split(' ')]

        list = objdict[subj]
        boxes = list['boxes']
        scores = list['scores']
        classes = list['classes']

        avgHead, avgGroin, avgHand = getAvgHeadAndGroin(boxes, scores, classes)
        head, groin, hands, cbs = getLabelsFromAnnotation(subj, frame, zonesTrue)
        if groin is None :
            groin = avgGroin


        for zone, box in cbs.iteritems():
            input = [float(frame)/boxes.shape[0]]
            input.extend(head)
            input.extend(groin)
            #input.extend((groin[0], groin[2]))
            input.extend(avgHand)
            input.extend(box)

            zonePred = predictZone(zoneModel, input)
            count += 1
            if zonePred != zone and zone not in [100]:
                print " Filename ", filename, " Incorrect prediction : ", zonePred, zone
                incorrectCount += 1

    print "Total incorrect = ", incorrectCount, " out of ", count

def generateLabels() :
    annotationFiles = []
    # Collect all annotated files.
    for filename in os.listdir(IOHandler.ANNOTATION_FOLDER):
        if filename.endswith(".xml") :
            annotationFiles.append(filename)

    zonedata = getLabels(annotationFiles)
    zonedata.to_csv(IOHandler.LABELED_PREDICTED_DATA, index=False, header=False)

def testAnnotationFiles():
    annotationFiles = []
    # Collect all annotated files.
    for filename in os.listdir(IOHandler.ANNOTATION_FOLDER):
        if filename.endswith(".xml") :
            annotationFiles.append(filename)
    testLabels(annotationFiles)

testAnnotationFiles()

# df = IOHandler.load_trainingLabels()
# count = 0
# for index, row in df.iterrows():
#     subj = row['Subject']
#     zoneStr = row['Zones']
#     zonesTrue = [] if not zoneStr else [int(k) for k in zoneStr.split(' ')]
#     if 12 in zonesTrue and 14 in zonesTrue :
#         if IOHandler.isInLabeledSet(subj) :
#             print "Subject : ", subj
#         count += 1
#
# print "Count = ", count