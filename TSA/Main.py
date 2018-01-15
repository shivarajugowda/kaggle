import numpy as np
import tensorflow as tf
import matplotlib
import IOHandler
import os
import pandas as pd

import sklearn.metrics as metrics
import matplotlib.pyplot as plt
import time
import pickle
from collections import Counter


import csv
import os
import scipy as sp

DATA_FORMAT = IOHandler.FORMAT_APS

MIN_SCORE_THRESHOLD = 0.05
MIN_NUM_FRAMES = 1

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
                if scores[i] > 0.8 :
                    hands.append(box)
            elif class_name == 'Groin':
                if scores[i] > 0.5 and scores[i] > groinMaxScore :
                    groinMaxScore = scores[i]
                    groin = box
            elif class_name == 'Contraband':
                contrabands.append((scores[i],box))
    return head, hands, groin, contrabands

def predictZone(model, X_test) :
    x = np.array(X_test)
    x = np.expand_dims(x, axis=0)
    # make predictions for test data
    y_pred = model.predict(x)
    predictions = [int(value) for value in y_pred]
    return predictions[0]

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

class CBBox:
    ndigits = 2
    def __init__(self,cbbox):
        self.ymin = round(cbbox[0], self.ndigits)
        self.ymax = round(cbbox[2], self.ndigits)

    def __hash__(self):
        return hash((self.ymin, self.ymax))

    def __eq__(self, other):
        return np.isclose(self.ymin, other.ymin) and \
               np.isclose(self.ymax, other.ymax)

    def __ne__(self, other):
        # Not strictly necessary, but to avoid having both x==y and x!=y
        # True at the same time
        return not(self == other)

# Returns number of frames a contraband is found for each of the zones.
def detectContrabands(df):
    pkl_file = open(IOHandler.DETECTED_OBJECTS_PKL_FILE, 'rb')
    objdict = pickle.load(pkl_file)
    pkl_file.close()

    timeZones = 0
    result = pd.DataFrame(columns=['Subject', 'Zone', 'NFrames'])
    zoneModel = IOHandler.getZoneDetectorModel()

    for index, row in df.iterrows():
        subj = row['Subject']
        if subj not in objdict :
            print "Didn't find ", subj, ' in the pkl file'
            continue

        #if subj not in ['346fc1c788ad7fb4da5a11f3bfb07d28'] :
        #    continue

        list = objdict[subj]
        boxes = list['boxes']
        scores = list['scores']
        classes = list['classes']

        zones = np.zeros(18)
        zonescores = np.zeros(18)
        zoneStr = row['Zones'] if 'Zones' in row else ""
        zonesTrue = [] if not zoneStr else [int(k) for k in str(zoneStr).split(' ')]
        avgHead, avgGroin, avgHand = getAvgHeadAndGroin(boxes, scores, classes)
        CBBoxZones = {}

        for frame in range(boxes.shape[0]) :
            head, hands, groin, contrabands = getObjectsInFrame(boxes[frame,:,:], scores[frame,:], classes[frame, :])
            cbZones = []
            if head is None:
                continue

            for cb in contrabands:
                cb_score = cb[0]
                cb_box = cb[1]
                hd = head if head is not None else avgHead
                gr = groin if groin is not None else avgGroin
                input = [float(frame)/boxes.shape[0]]
                input.extend(hd)
                input.extend(avgGroin)
                #input.extend((gr[0], gr[2]))
                input.extend(avgHand)
                input.extend(cb_box)

                zone = predictZone(zoneModel, input)
                if  zone > 0 and zone not in cbZones:
                    zones[zone] += 1
                    zonescores[zone] += cb_score
                    cbZones.append(zone)
                    cbbox = CBBox(cb_box)

                    if cbbox in CBBoxZones:
                        CBBoxZones[cbbox].append(zone)
                    else:
                        CBBoxZones[cbbox] = [zone]
                    #if zone not in zonesTrue:
                        #visualizeFrame(subj, frame)
                        #saveFrame(subj, frame, zone, row[2], head, hands, groin, contrabands)


        # for i in range(1,zones.shape[0]) :
        #    result.loc[len(result)] = [subj, i, zones[i]]

        zones = np.zeros(18)
        #print "The size of CBBoxZones = ", len(CBBoxZones)
        for k, v in CBBoxZones.iteritems():
            freqZone = Counter(v).most_common(1)[0][0]
            #print k.box[0], k.box[2], " : ",  v, " Most Common = ", freqZone
            zones[freqZone] += len(v)

        for i in range(1, zones.shape[0]):
            if  zones[i] == 1 and zonescores[i] < 0.6 :
                zones[i] = 0

        for i in range(1,zones.shape[0]) :
            result.loc[len(result)] = [subj, i, zones[i]]

    return  result

def predict(df) :
    template = "{0:32}  {1:3}  {2:>10}  {3:>15}  {4:>10}  {5:>12} {6:>12}"  # column widths: 8, 10, 15, 7, 10
    print template.format("Subject", "#CB", "Zones", "Prediction", "NumCorrect", "NumInCorrect", "Missing")  # header

    result = detectContrabands(df)

    totCb, totCorrect, totIncorrect, totMissing = 0, 0, 0, 0
    inCorrectZones, missingZones = np.zeros(18), np.zeros(18)

    for index, row in df.iterrows():
        subj = row['Subject']
        golden = row['Zones'] if 'Zones' in row else None
        zonesG = [int(k) for k in golden.split(' ')] if golden else []
        zonesG.sort()
        ncb = len(zonesG)

        numCorrect, numInCorrect, numMissing = 0, 0, 0
        predicitons = result[(result['Subject'] == subj) & (result['NFrames'] >= MIN_NUM_FRAMES)]['Zone'].tolist()

        for p in predicitons :
            if p in zonesG :
                numCorrect += 1
            else :
                numInCorrect += 1
                inCorrectZones[p] += 1

        for z in zonesG :
            if z not in predicitons :
                numMissing += 1
                missingZones[z] += 1
                #if z == 5 :
                #generateImage(subj, None)
                #print "Subj = ", subj, " Predicted ", predicitons , " Actual : ", zonesG, " Missing : ",  z

        totCb += ncb
        totCorrect += numCorrect
        totIncorrect += numInCorrect
        totMissing += numMissing

        if numMissing >  0 :
            print template.format(subj,
                                  ncb,
                                  ' '.join(str(a) for a in zonesG),
                                  ' '.join(str(a) for a in predicitons),
                                  numCorrect,
                                  numInCorrect,
                                  numMissing)
            #print result[(result['Subject'] == subj)]['NFrames'].tolist()
            #visualize(subj)
            generateImage(subj, None)

    print "-" * 105
    print template.format("Total",
                          totCb,
                          ' ',
                          ' ',
                          totCorrect,
                          totIncorrect,
                          totMissing)

    print "\nAggregated Zone inconsistencies :"
    df = pd.DataFrame({"Zone":np.arange(1,18), "Incorrect":inCorrectZones[1:], "Missing": missingZones[1:]})
    df.sort_values('Incorrect', ascending=False, inplace=True)
    #print df[['Zone', 'Incorrect', 'Missing']].to_string(index=False)
    return result


def saveFrame(subj, frame, pred, actual, head, hands, groin, contrabands) :
    print "Subj = ", subj, " Frame = ", frame, " Predicted ", pred, " Actual : ", actual
    IOHandler.save_frame(IOHandler.INCORRECT_IMAGES_FOLDER, subj, frame, DATA_FORMAT)
    IOHandler.generateAnnotation(IOHandler.INCORRECT_IMAGES_FOLDER, subj, frame, head, hands, groin, contrabands)

def generateImage(subj, objdict) :

    # If the subj already part of the test set, do nothing.
    #if IOHandler.isInLabeledSet(subj) :
    #    return

    if objdict is None :
        pkl_file = open(IOHandler.DETECTED_OBJECTS_PKL_FILE, 'rb')
        objdict = pickle.load(pkl_file)
        pkl_file.close()

    list = objdict[subj]
    boxes = list['boxes']
    scores = list['scores']
    classes = list['classes']

    IOHandler.save_images(IOHandler.INCORRECT_IMAGES_FOLDER, subj, DATA_FORMAT)
    for frame in range(boxes.shape[0]):
        head, hands, groin, contrabands = getObjectsInFrame(boxes[frame, :, :], scores[frame, :], classes[frame, :])
        IOHandler.generateAnnotation(IOHandler.INCORRECT_IMAGES_FOLDER, subj, frame, head, hands, groin, contrabands)


def genImagesInBulk(df):
    pkl_file = open(IOHandler.DETECTED_OBJECTS_PKL_FILE, 'rb')
    objdict = pickle.load(pkl_file)
    pkl_file.close()

    count = 0
    for index, row in df.iterrows():
        subj = row['Subject']
        print "Count = ", count, "Working on ", subj
        count += 1
        generateImage(subj, objdict)


def logLoss(df) :
    result = detectContrabands(df)
    y_act = []
    y_pred = result['NFrames'].map(getProbability)

    for index, row in df.iterrows():
        subj = row['Subject']
        zoneStr = row['Zones']
        zonesG = [] if not zoneStr else [int(k) for k in zoneStr.split(' ')]
        for i in range(1,18) :
            if i in zonesG :
                y_act.append(1.0)
            else :
                y_act.append(0.0)

    print "Log loss = ", metrics.log_loss(y_act, y_pred)


def logLossV2(df) :
    result = detectContrabands(df)

    scores = []
    for index, row in df.iterrows():
        subj = row['Subject']
        zoneStr = row['Zones']
        y_act = []
        zonesG = [] if not zoneStr else [int(k) for k in zoneStr.split(' ')]
        for i in range(1,18) :
            if i in zonesG :
                y_act.append(1.0)
            else :
                y_act.append(0.0)

        resSubj = result[result['Subject'] == subj]
        y_pred = resSubj['NFrames'].map(getProbability)
        loss = llfun(y_act, y_pred)
        scores.append(loss)

    print "Log loss = ", (sum(scores) / len(scores))


def llfun(act, pred):
    epsilon = 1e-15
    pred = sp.maximum(epsilon, pred)
    pred = sp.minimum(1-epsilon, pred)
    ll = sum(act*sp.log(pred) + sp.subtract(1,act)*sp.log(sp.subtract(1,pred)))
    ll = ll * -1.0/len(act)
    return ll

def getProbability(nFrames) :
    if nFrames >= 4:
        return 0.9908
    elif nFrames == 3:
        return 0.9174
    elif nFrames == 2:
        return 0.6157
    elif nFrames == 1:
        return 0.2135
    else:
        return 0.00184

    if nFrames >= 3 :
        return 0.96
    elif nFrames == 2 :
        return 0.7894
    elif nFrames == 1 :
         return 0.214
    else :
        return 6/(200*17.0 - (294+30))

def submit(df) :
    result = detectContrabands(df)
    result['Id'] = result['Subject'] + '_Zone' + result['Zone'].map(str)
    result['Probability'] = result['NFrames'].map(getProbability)
    result.sort_values("Id", inplace=True)
    result[['Id', 'Probability']].to_csv('out.csv', index=False)


start_time = time.time()
#df = IOHandler.load_trainingLabels()
#df = df[df['NumContrabands'].isin([0])]
#df = df[df['NumContrabands'] == 3].sample(200, random_state=10)
#df.sort_values("Subject", inplace=True)
# df = df.head(100)
#df = df[df['NumContrabands'] == 1][0:100] #.head(100)

#subjects = ['0050492f92e22eed3474ae3a6fc907fa', '9ec808303497389de113d609a65c7935',
#                        'e36ec0125714375592191df6eac2517c']
#df = df[df.Subject.isin(subjects)]

#print df
#detectAndSave(df['Subject'].tolist())

#df = IOHandler.load_trainingLabels()
#df = IOHandler.load_testSet()
#df.sort_values("Subject", inplace=True)
#prds = predict(df)
#Profile.run('predict(df)')


#df = IOHandler.load_solutionSet()
#df = df.sample(100, random_state=32)
#df.sort_values("Subject", inplace=True)
#prds = predict(df)
#logLoss(df)

df = IOHandler.load_submitSet()
submit(df)
#genImagesInBulk(df)
#genImages(df)
#visualize()

print("Total time taken : %s seconds" % (time.time() - start_time))