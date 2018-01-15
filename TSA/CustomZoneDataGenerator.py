import numpy as np
import tensorflow as tf
import matplotlib
import IOHandler
import os
import pandas as pd
from collections import Counter

import sklearn.metrics as metrics
import pickle


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

def getCenter(point) :
    x = (point[0] + point[2]) / 2.0
    y = (point[1] + point[3]) / 2.0
    return (x,y)

def getGroinZone(frame, head, groin, cb) :
    p = getCenter(cb)

    if groin is not None :
        if groin[0] <= cb[0] <= groin[2] and groin[1] <= cb[1] <= cb[3] <= groin[3] :
            return 9

        # If facing forward or backward
        if frame in [2, 1, 0, 15, 14] :
            if groin[0] <= cb[0] <= groin[2] and groin[1] <= p[1] <= groin[3] :
                return 9

    return 0

def getHipZone(frame, head, groin, cb):
    rect = groin if groin is not None else head
    ctr = getCenter(rect)
    cbCtr = getCenter(cb)

    side = None
    if 4 < frame < 12 :
        side = 'Left' if cbCtr[1] < ctr[1] else 'Right'
    else :
        side = 'Right' if cbCtr[1] < ctr[1] else 'Left'

    # Handle Cusp Cases
    if frame in [2, 3]:
        side = 'Middle' if cbCtr[1] < ctr[1] else 'Left'
    elif frame == 4:
        side = 'Left'
    elif frame == 5:
        side = 'Left' if cbCtr[1] < rect[3] else 'Right'
    elif frame == 11:
        side = 'Left' if cbCtr[1] < rect[1] else 'Right'
    elif frame == 12:
        side = 'Right'
    elif frame in [13,14]:
        side = 'Right' if cbCtr[1] < ctr[1] else 'Middle'


    hipValues = {'Right':8, 'Middle':9, 'Left':10}
    return hipValues[side]

def getThighZone(frame, head, groin, cb):
    rect = groin if groin is not None else head
    ctr = getCenter(rect)
    cbCtr = getCenter(cb)

    side = None
    if 4 < frame < 12 :
        side = 'Left' if cbCtr[1] < ctr[1] else 'Right'
    else :
        side = 'Right' if cbCtr[1] < ctr[1] else 'Left'

    # Handle Cusp Cases
    if frame == 4:
        side = 'Left'
    elif frame == 5:
        side = 'Left' if cbCtr[1] < rect[3] else 'Right'
    elif frame == 11:
        side = 'Left' if cbCtr[1] < rect[1] else 'Right'
    elif frame == 12:
        side = 'Right'


    valKeys = {'Right':11, 'Left':12}
    return valKeys[side]


def getGroinSide(frame, head, groin, cb) :
    rect = groin if groin is not None else head
    ctr = getCenter(rect)
    cbCtr = getCenter(cb)

    if frame == 13 :
        return 'Right' if cbCtr[1] < rect[1] else 'Left'

    #if frame == 3 :
    #    return 'Right' if cbCtr[1] < ctr[1] else 'Left'

    if frame in [4] :
        return 'Left'
    elif frame in [12] :
        return  'Right'
    elif frame in [2, 3] :
        return 'Right' if cbCtr[1] < rect[3] else 'Left'
    elif frame == 11 :
        return 'Left' if cbCtr[1] < rect[1] else 'Right'
    elif frame == 5 :
         return 'Left' if cbCtr[1] < rect[3] else 'Right'


    if 4 < frame < 12 :
        return 'Left' if cbCtr[1] < ctr[1] else 'Right'
    else :
        return 'Right' if cbCtr[1] < ctr[1] else 'Left'

def getHeadSide(frame, head, cb) :
    cbCtr = getCenter(cb)
    headLen = head[3] - head[1]

    ctr = head[1] # Zones 3, 11
    if frame in [2,10] :
        ctr += (1.0/6.0) * headLen
    elif frame in [1, 9] :
        ctr += (2.0 / 6.0) * headLen
    elif frame in [0, 8] :
        ctr += (1.0/6.0) * headLen
    elif frame in [7, 15] :
        ctr += (4.0 / 6.0) * headLen
    elif frame in [6, 14]:
        ctr += (5.0 / 6.0) * headLen
    elif frame in [5, 13]:
        ctr = head[3]

    if frame is 4 :
        return 'Left'
    elif frame is 12 :
        return 'Right'

    if 4 < frame < 12 :
        return 'Left' if cbCtr[1] < ctr else 'Right'
    else :
        return 'Right' if cbCtr[1] < ctr else 'Left'


def getFrontBack(frame, head, cb) :
    ctr = getCenter(head)
    cbCtr = getCenter(cb)

    if frame in [3, 4, 5, 11, 12, 13] :
        return None

    if frame == 4:
        return 'Front' if cbCtr[1] < head[3] else 'Back'
    if frame == 12:
        return 'Back' if cbCtr[1] < head[1] else 'Front'

    if 4 < frame < 12 :
        return 'Back'
    else :
        return 'Front'


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

def predictZone(subj, frame, head, groin, contraband, avgHead, avgGroin) :

    # if frame not in [2] :
    #     return 0

    if contraband is None :
        return 0

    cbCtr = getCenter(contraband)

    # Special case for groin.
    zone = getGroinZone(frame, head, groin, contraband)
    if zone != 0 :
        return zone

    if head == None :
        head = avgHead

    headCtr = getCenter(head)

    hdSide = getHeadSide(frame, head, contraband)
    frontOrBack = getFrontBack(frame, head, contraband)

    # Forearm is above the mid head.
    foreArmCutOff = headCtr[0]
    if cbCtr[0] < foreArmCutOff :
        return 2 if hdSide == 'Right' else 4

    # UPPER BODY.
    upperBodyStart = head[2]
    upperBodyEnd = avgGroin[0] if groin is None else groin[0]
    upperBodyLength = upperBodyEnd - upperBodyStart

    # UpperArm
    if foreArmCutOff < cbCtr[0] < upperBodyStart :
        return 1 if hdSide == 'Right' else 3

    if foreArmCutOff < cbCtr[0] < upperBodyStart + 0.30*upperBodyLength  \
            and (contraband[3] < head[1] or contraband[1] > head[3]):
        return 1 if hdSide == 'Right' else 3

    if frontOrBack is not None and upperBodyStart < cbCtr[0] < (upperBodyStart + 0.35*upperBodyLength) :
        # Chest and Upper Back
        return 5 if frontOrBack == 'Front' else 17
    elif upperBodyStart < cbCtr[0] < upperBodyEnd :
        # Torso
        return 6 if hdSide == 'Right' else 7

    # LOWER BODY.
    grSide = getGroinSide(frame, head, groin, contraband)
    lowerBodyStart = avgGroin[0]
    lowerLength = 1.0 - lowerBodyStart

    if lowerBodyStart < cbCtr[0] < (lowerBodyStart + 0.32*lowerLength) :
        return getHipZone(frame, head, groin, contraband)
    elif lowerBodyStart < cbCtr[0] < (lowerBodyStart + 0.50*lowerLength) :
        return getThighZone(frame, head, groin, contraband)
    elif lowerBodyStart < cbCtr[0] < (lowerBodyStart + 0.75*lowerLength) :
        return 13 if grSide == 'Right' else 14
    elif lowerBodyStart < cbCtr[0] :
        return 15 if grSide == 'Right' else 16

    return 0


def doesOverlap(rect1, rect2) :
    diff1 = rect1[2] - rect2[0]
    diff2 = rect2[2] - rect1[0]
    if diff1 >=0 and diff2  >= 0 :
        overlap = min(diff1, diff2)
        if overlap >= 0.8*(rect1[2]-rect1[0]) and overlap >= 0.8*(rect2[2]-rect2[0]) :
            return True

    return False

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

def getOppositeZone(zone):
    oppositeZones = {1:[3], 2:[4], 3:[1], 4:[2], 6:[7], 7:[6],
                     5:[], 17:[],
                     8:[9,10], 9:[8,10], 10:[9,8],
                     11:[12], 12:[11], 13:[14], 14:[13],
                     15:[16], 16:[15]}
    return oppositeZones[zone]

def predictZoneModel(model,X_test) :
    x = np.array(X_test)
    x = np.expand_dims(x, axis=0)
    # make predictions for test data
    y_pred = model.predict(x)
    predictions = [int(value) for value in y_pred]
    return predictions[0]

# Returns number of frames a contraband is found for each of the zones.
def detectContrabands(df):
    pkl_file = open(IOHandler.DETECTED_OBJECTS_PKL_FILE, 'rb')
    objdict = pickle.load(pkl_file)
    pkl_file.close()

    #zoneModel = IOHandler.getZoneDetectorModel()
    result = pd.DataFrame(columns=['Subject', 'Zone', 'NFrames'])
    zonePreds = pd.DataFrame(columns=['Frame',
                                         'Head_0', 'Head_1', 'Head_2', 'Head_3',
                                         'Groin_1', 'Groin_2', 'Groin_3', 'Groin_4',
                                         'Hand_0', 'Hand_1',
                                         'CB_1', 'CB_2', 'CB_3', 'CB_4',
                                         'Zone'
                                         ])

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

        CBBoxZones = {}
        zonescores = np.zeros(18)
        zones = np.zeros(18)
        zoneStr = row['Zones'] if 'Zones' in row else ""
        zonesTrue = [] if not zoneStr else [int(k) for k in zoneStr.split(' ')]
        avgHead, avgGroin, avgHand = getAvgHeadAndGroin(boxes, scores, classes)
        for frame in range(boxes.shape[0]) :
            head, hands, groin, contrabands = getObjectsInFrame(boxes[frame,:,:], scores[frame,:], classes[frame, :])
            gr = groin if groin is not None else avgGroin
            cbZones = []
            for cb in contrabands:
                cb_score = cb[0]
                cb_box = cb[1]

                # input = [float(frame) / boxes.shape[0]]
                # input.extend(head)
                # input.extend(gr)
                # # input.extend((groin[0], groin[2]))
                # input.extend(avgHand)
                # input.extend(cb_box)
                # zoneM = predictZoneModel(zoneModel, input)

                zone = predictZone(subj, frame, head, groin, cb_box, avgHead, avgGroin)
                if  zone > 0 and zone not in cbZones:
                    zones[zone] += 1
                    zonescores[zone] += cb_score
                    cbZones.append(zone)

                    cbbox = CBBox(cb_box)
                    if cbbox in CBBoxZones:
                        CBBoxZones[cbbox].append(zone)
                    else:
                        CBBoxZones[cbbox] = [zone]

                    # if zone not in zonesTrue :
                    #     print "Subj = ", subj, " Frame = ", frame, " Predicted ", zone, " Actual : ", row[2]
                    #     visualizeFrame(subj, frame)
                    if zone in zonesTrue and head is not None\
                            and not \
                            (frame in [3, 4, 5, 11, 12, 13] and
                             any(x in getOppositeZone(zone)  for x in zonesTrue) ) :
                        input = [float(frame)/boxes.shape[0]]
                        input.extend(head)
                        input.extend(gr)
                        # input.extend((groin[0], groin[2]))
                        input.extend(avgHand)
                        input.extend(cb_box)
                        input.append(zone)
                        zonePreds.loc[len(zonePreds)] = input


        zones = np.zeros(18)
        for k, v in CBBoxZones.iteritems():
            freqZone = Counter(v).most_common(1)[0][0]
            #print k.box[0], k.box[2], " : ",  v, " Most Common = ", freqZone
            zones[freqZone] += len(v)

        for i in range(1, zones.shape[0]):
            if zones[i] == 1 and zonescores[i] < 0.6:
                zones[i] = 0

        for i in range(1,zones.shape[0]) :
            result.loc[len(result)] = [subj, i, zones[i]]

    #zonePreds.to_csv(IOHandler.CUSTOM_PREDICTED_DATA, index=False, header=False)
    return  result

def predict(df) :
    template = "{0:32}  {1:3}  {2:>10}  {3:>15}  {4:>10}  {5:>12}"  # column widths: 8, 10, 15, 7, 10
    print template.format("Subject", "#CB", "Zones", "Prediction", "NumCorrect", "NumInCorrect")  # header

    result = detectContrabands(df)
    #resultHighThrs = detectContrabands(df)
    totCb, totCorrect, totIncorrect = 0, 0, 0

    analysis = pd.DataFrame(columns=['TrueValue', 'Predicted'])

    for index, row in df.iterrows():
        subj = row['Subject']
        golden = row['Zones'] if 'Zones' in row else None
        zonesG = [int(k) for k in golden.split(' ')] if golden else []
        zonesG.sort()
        ncb = len(zonesG)

        #if subj not in ['346fc1c788ad7fb4da5a11f3bfb07d28'] :
        #    continue

        predicitons = result[(result['Subject'] == subj) & (result['NFrames'] >= MIN_NUM_FRAMES)]['Zone'].tolist()
        #preds2 = resultHighThrs[(resultHighThrs['Subject'] == subj) & (resultHighThrs['NFrames'] >= 3)]['Zone'].tolist()
        #predicitons.extend(x for x in preds2 if x not in predicitons)

        numCorrect = sum(p in zonesG for p in predicitons)
        numInCorrect = len(predicitons) - numCorrect

        totCb += ncb
        totCorrect += numCorrect
        totIncorrect += numInCorrect

        if not predicitons :
            analysis.loc[len(analysis)] = [zonesG, 0]
            # print template.format(subj,
            #                       ncb,
            #                       ' '.join(str(a) for a in zonesG),
            #                       ' '.join(str(a) for a in predicitons),
            #                       numCorrect,
            #                       numInCorrect)
            # print result[(result['Subject'] == subj)]['NFrames'].tolist()
            # visualize(subj)
        else :
            for pred in predicitons :
                analysis.loc[len(analysis)] = [zonesG, pred]

        if numInCorrect > 0 :
            print template.format(subj,
                                  ncb,
                                  ' '.join(str(a) for a in zonesG),
                                  ' '.join(str(a) for a in predicitons),
                                  numCorrect,
                                  numInCorrect)
            #print result[(result['Subject'] == subj)]['NFrames'].tolist()
            #visualize(subj)

    print "-" * 92
    print template.format("Total",
                          totCb,
                          ' ',
                          ' ',
                          totCorrect,
                          totIncorrect)


    # print "Total correctly predicted = "
    # incorr = analysis[analysis['TrueValue'] != analysis['Predicted']]
    # with pd.option_context('display.max_rows', None, 'display.max_columns', 3):
    #     print analysis.groupby(['TrueValue', 'Predicted']).size()
    return result


def logLoss(df) :
    result = detectContrabands(df)
    y_act = []
    y_pred = result['NFrames'].map(getProbability)

    for index, row in df.iterrows():
        subj = row['Subject']
        zonesStr = row['Zones']
        zonesG = [] if not zonesStr else [int(k) for k in zonesStr.split(' ')]
        for i in range(1,18) :
            if i in zonesG :
                y_act.append(1.0)
            else :
                y_act.append(0.0)

    print "Log loss = ", metrics.log_loss(y_act, y_pred)

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
        return 0.005


def submit(df) :
    result = detectContrabands(df)
    result['Id'] = result['Subject'] + '_Zone' + result['Zone'].map(str)
    result['Probability'] = result['NFrames'].map(getProbability)
    result.sort_values("Id", inplace=True)
    result[['Id', 'Probability']].to_csv('out.csv', index=False)

#df = IOHandler.load_testingSet()

#df = df[df['NumContrabands'] >= 3].sample(300, random_state=10)
#df = df[df['NumContrabands'] == 1][0:100] #.head(100)

# subjects = ['d904d73f5e53eed05fef89ce0032fc1c', 'cbc5410d54d2ff5f3b3b818c9589e53c',
#                         '7a465515247d5150a437499ed4dd31a8', '9657d70069ba334ec5e7dad5aa189aea',
#                         '1603eddceceddc29f790fcf5ba04bec8', '7828d838474e3b54306a164d1ba6419d',
#                         '158c596536e90a3edb3c84dbecc29420']
#detectAndSave(subjects)

#print df
#detectAndSave(df['Subject'].tolist())

#df = IOHandler.load_trainingLabels()
#df = df[df['NumContrabands'].isin([1,2,3])]
#prds = predict(df)

#df = IOHandler.load_trainingLabels()
#prds = predict(df)
#logLoss(df)

df = IOHandler.load_submitSet()
submit(df)

#genImages(df)
#visualize()
