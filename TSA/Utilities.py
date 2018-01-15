import IOHandler
import os, io, base64, json
from lxml import etree
from shutil import copyfile
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import pandas as pd

# Scale annotation 293 x 376 to 512 x 660
def _scaleAnnotation(fileName) :
    xml = etree.parse(fileName)
    xml.xpath("//annotation/folder")[0].text = "images"
    xml.xpath("//annotation/size/width")[0].text = "512"
    xml.xpath("//annotation/size/height")[0].text = "660"

    for elem in xml.iter("object"):
        name = elem.xpath("./name/text()")[0]

        xmin = float(elem.xpath("./bndbox/xmin/text()")[0])*512/293
        ymin = float(elem.xpath("./bndbox/ymin/text()")[0])*660/376
        xmax = float(elem.xpath("./bndbox/xmax/text()")[0])*512/293
        ymax = float(elem.xpath("./bndbox/ymax/text()")[0])*660/376

        elem.xpath("./bndbox/xmin")[0].text = str(int(xmin))
        elem.xpath("./bndbox/ymin")[0].text = str(int(ymin))
        elem.xpath("./bndbox/xmax")[0].text = str(int(xmax))
        elem.xpath("./bndbox/ymax")[0].text = str(int(ymax))

    return xml

def scaleImages(origFolder, modFolder) :
    count = 0
    origAnnotationFolder = IOHandler.getAnnotationFolder(origFolder)
    modAnnotationFolder  = IOHandler.getAnnotationFolder(modFolder)
    for filename in os.listdir(origAnnotationFolder):
        if not filename.endswith(".xml"):
            continue
        subj, frame = IOHandler.getSubjAndFrameFromFileName(filename)

        IOHandler.save_frame(modFolder, subj, frame, IOHandler.FORMAT_APS)
        xml = _scaleAnnotation(os.path.join(origAnnotationFolder, filename))
        xml.write(os.path.join(modAnnotationFolder, filename), pretty_print=True)
        count += 1
        #if count > 20:
        #    break

def copyImagesWithZones(origFolder, modFolder) :
    count = 0
    origAnnotationFolder = IOHandler.getAnnotationFolder(origFolder)
    modAnnotationFolder = IOHandler.getAnnotationFolder(modFolder)
    if not os.path.exists(modAnnotationFolder):
        os.makedirs(modAnnotationFolder)
    for filename in os.listdir(origAnnotationFolder):
        if filename.endswith(".xml") and \
            'Zone_' in open(os.path.join(origAnnotationFolder, filename)).read() :
            copyfile(os.path.join(origAnnotationFolder, filename), os.path.join(modAnnotationFolder, filename))
            imgFileName = filename.replace(".xml", ".jpg")
            copyfile(os.path.join(origFolder, imgFileName), os.path.join(modFolder, imgFileName))
            count += 1

def images2JSON() :
    subjectsDict = {}
    #df = IOHandler.load_trainingLabels()
    # df = df[df['NumContrabands'] == 3].sample(300, random_state=10)
    # df.sort_values("Subject", inplace=True)
    # df = df.head(3)
    #trainList = df['Subject'].tolist()
    testList = IOHandler.load_submitSet()['Subject'].tolist()
    #subjList = trainList + testList
    subjList = testList
    count = 0
    for subj in subjList:
        print "Count = ", count, " Working on ", subj
        count += 1
        data = IOHandler.read_data(subj, IOHandler.FORMAT_APS)
        frames = []
        for frame in range(0, data.shape[2]):
            img_np = np.flipud(data[:, :, frame].transpose())
            img_np *= 1.0 / img_np.max()  # Normalize image
            img_np = np.uint8(plt.cm.viridis(img_np) * 255)  # Apply colormap
            img_np = img_np[:, :, :3]

            img = Image.fromarray(img_np)
            output_str = io.BytesIO()
            img.save(output_str, "JPEG")
            frames.append(base64.b64encode(output_str.getvalue()))
            output_str.close()
        subjectsDict[subj] = frames


    with open(IOHandler.INPUTS_JSON, 'w') as fp:
        json.dump(subjectsDict, fp)


def copyImages(origFolder, modFolder) :
    count = 0
    origAnnotationFolder = IOHandler.getAnnotationFolder(origFolder)
    modAnnotationFolder = IOHandler.getAnnotationFolder(modFolder)
    if not os.path.exists(modAnnotationFolder):
        os.makedirs(modAnnotationFolder)
    for filename in os.listdir(origAnnotationFolder):
        if filename.endswith(".xml") and \
            'Head' in open(os.path.join(origAnnotationFolder, filename)).read() :
            copyfile(os.path.join(origAnnotationFolder, filename), os.path.join(modAnnotationFolder, filename))
            imgFileName = filename.replace(".xml", ".jpg")
            copyfile(os.path.join(origFolder, imgFileName), os.path.join(modFolder, imgFileName))
            count += 1

def removeContrabandsFromImages(origFolder, modFolder):
    count = 0
    origAnnotationFolder = IOHandler.getAnnotationFolder(origFolder)
    modAnnotationFolder = IOHandler.getAnnotationFolder(modFolder)
    if not os.path.exists(modAnnotationFolder):
        os.makedirs(modAnnotationFolder)
    for filename in os.listdir(origAnnotationFolder):
        if not filename.endswith(".xml"):
            continue

        xml = etree.parse(os.path.join(origAnnotationFolder, filename))
        # Remove contraband element
        for elem in xml.iter("object"):
            name = elem.xpath("./name/text()")[0]
            if name == 'Contraband':
                elem.getparent().remove(elem)
        xml.write(os.path.join(modAnnotationFolder, filename), pretty_print=True)

        imgFileName = filename.replace(".xml", ".jpg")
        copyfile(os.path.join(origFolder, imgFileName), os.path.join(modFolder, imgFileName))
        count += 1

def createTestSet() :
    df = IOHandler.load_trainingLabels()
    df = df[~df['Subject'].map(IOHandler.isInLabeledSet)]
    df3 = df[df['NumContrabands'].isin([3])].sample(50, random_state=22)
    df2 = df[df['NumContrabands'].isin([2])].sample(50, random_state=22)
    df1 = df[df['NumContrabands'].isin([1])].sample(50, random_state=22)
    df0 = df[df['NumContrabands'].isin([0])].sample(50, random_state=22)
    result = pd.concat([df3, df2, df1, df0], ignore_index=True)
    df = result.sample(frac=1).reset_index(drop=True)
    df.to_csv('testset.csv', index=False)

#scaleImages(IOHandler.IMAGES_FOLDER, "images_mod")

#copyImagesWithZones("incorrect_images", "tmpImages")

#images2JSON()

#removeContrabandsFromImages("incorrect_images", "tmpImages")

#print round(2.1234, 1)

#createTestSet()

#copyImagesWithZones("incorrect_images", "tmpImages")
