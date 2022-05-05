#Part of the standard library 
import xml.etree.ElementTree as ET
from xml.dom import minidom
import os
import re
import random

#Not part of the standard library
import numpy as np 
import pandas as pd
import cv2
import dlib


def read_csv(input):
    '''
    This function reads a XY coordinate file (following the tpsDig coordinate system) containing several specimens(rows) 
    and any number of landmarks. It is generally assumed here that the file contains a header and no other 
    columns other than an id column (first column) and the X0 Y0 ...Xn Yn coordinates for n landmarks.It is also 
    assumed that the file contains no missing values.
        
    Parameters:
        input (str): The XY coordinate file (csv format)
    Returns:
        dict: dictionary containing two keys (im= image id, coords= array with 2D coordinates)
    '''
    csv_file = open(input, 'r') 
    csv =csv_file.read().splitlines()
    csv_file.close()
    im, coords_array = [], []
    for i, ln in enumerate(csv):
        if i > 0:
            im.append(ln.split(',')[0])
            coord_vec=ln.split(',')[1:]
            coords_mat = np.reshape(coord_vec, (int(len(coord_vec)/2),2))
            coords = np.array(coords_mat, dtype=float)
            coords_array.append(coords)
    return {'im': im, 'coords': coords_array}

def split_train_test(input, percentage = 0.8):
    '''
    Splits an image directory into 'train' and 'test' directories. The original image directory is preserved. 
    When creating the new directories, this function converts all image files to 'jpg'. The function returns
    a dictionary containing the image dimensions in the 'train' and 'test' directories.
    
    Parameters:
        input(dict): dictionary containing the image filenames and coordinates
        
    Returns:
        sizes (dict): dictionary containing the image dimensions in the 'train' and 'test' directories.
    '''
    # Listing the filenames.Folders must contain only image files (extension can vary).Hidden files are ignored
    random.seed(845)
    length = len(input['im'])
    l = list(range(length))
    random.shuffle(l)
    split = int(percentage * length)
    train_list = l[:split]
    test_list = l[split:]
    train_set = {'im': [input['im'][l] for l in train_list], 'coords': [input['coords'][l] for l in train_list]}
    test_set = {'im': [input['im'][l] for l in test_list], 'coords': [input['coords'][l] for l in test_list]}
    return train_set, test_set

def add_part_element(bbox,num):
    '''
    Internal function used by generate_dlib_xml. It creates a 'part' xml element containing the XY coordinates
    of an arbitrary number of landmarks. Parts are nested within boxes.
    
    Parameters:
        bbox (array): XY coordinates for a specific landmark
        num(int)=landmark id
        sz (int)=the image file's height in pixels
        
        
    Returns:
        part (xml tag): xml element containing the 2D coordinates for a specific landmark id(num)
    
    '''
    part = ET.Element('part')
    part.set('name',str(int(num)))
    part.set('x',str(int(bbox[0])))
    part.set('y',str(int(bbox[1])))
    return part

def add_bbox_element(bbox,imgtuple):
    '''
    Internal function used by generate_dlib_xml. It creates a 'bounding box' xml element containing the 
    four parameters that define the bounding box (top,left, width, height) based on the minimum and maximum XY 
    coordinates of its parts(i.e.,landmarks). An optional padding can be added to the bounding box.Boxes are 
    nested within images.
    
    Parameters:
        bbox (array): XY coordinates for all landmarks within a bounding box
        imgtuple (tuple): the image file's height and width in pixels
       
       
    Returns:
        box (xml tag): xml element containing the parameters that define a bounding box and its corresponding parts
    
    '''
    
    box = ET.Element('box')
    
    if len(imgtuple) == 2:
        height = imgtuple[0] - 2
        width = imgtuple[1] - 2
        top = 1
        left = 1
    elif len(imgtuple) == 4:
        left, top, width, height = imgtuple

    box.set('top', str(int(top)))
    box.set('left', str(int(left)))
    box.set('width', str(int(width)))
    box.set('height', str(int(height)))
    for i in range(0,len(bbox)):
        box.append(add_part_element(bbox[i,:],i))
    return box

def add_image_element(coords, imgtuple, path):
    '''
    Internal function used by generate_dlib_xml. It creates a 'image' xml element containing the 
    image filename and its corresponding bounding boxes and parts. 
    
    Parameters:
        image (str): image filename
        coords (array)=  XY coordinates for all landmarks within a bounding box
        sz (int)= the image file's height in pixels
        
        
    Returns:
        image (xml tag): xml element containing the parameters that define each image (boxes+parts)
    
    '''
    image_e = ET.Element('image')
    image_e.set('file', str(path))
    image_e.append(add_bbox_element(coords, imgtuple))
    return image_e

def generate_dlib_xml(image_dict, workdir: str, out_file):
    '''
    Generates a dlib format xml file for training or testing of machine learning models. 
    
    Parameters:
        images (dict): dictionary output by read_tps or read_csv functions 
        sizes (dict)= dictionary of image file sizes output by the split_train_test function
        folder(str)= name of the folder containing the images 
        
        
    Returns:
        None (file written to disk)
    '''
    root = ET.Element('dataset')
    root.append(ET.Element('name'))
    root.append(ET.Element('comment'))

    images_e = ET.Element('images')
    root.append(images_e)

    for i, imgfile in enumerate(image_dict['im']):
        path=os.path.join(workdir,'images', imgfile)
        if os.path.isfile(path) is True:
            img = cv2.imread(path)
            height, width = img.shape[0],img.shape[1]
            present_tags=[]
            for img in images_e.findall('image'): #need to simplify this
                present_tags.append(img.get('file'))   

            if path in present_tags:
                pos=present_tags.index(path)
                images_e[pos].append(add_bbox_element(image_dict['coords'][i],(height,width)))

            else:    
                images_e.append(add_image_element(image_dict['coords'][i], (height,width), path))

    et = ET.ElementTree(root)
    xmlstr = minidom.parseString(ET.tostring(et.getroot())).toprettyxml(indent="   ")
    with open(out_file, "w") as f:
        f.write(xmlstr)


def predictions_to_xml(predictor_name, dir, ignore, out_file):
    '''
    Generates a dlib format xml file for model predictions. It uses previously trained models to
    identify objects in images and to predict their shape. 
    
    Parameters:
        predictor_name (str): shape predictor filename
        dir(str): name of the directory containing images to be predicted
        out_file (str): name of the output file (xml format)
        
    Returns:
        None (out_file written to disk)
    
    '''
    extensions = {'.jpg', '.JPG', '.jpeg', '.JPEG', '.tif','.TIF'}
    predictor = dlib.shape_predictor(predictor_name)
    root = ET.Element('dataset')
    root.append(ET.Element('name'))
    root.append(ET.Element('comment'))
    images_e = ET.Element('images')
    root.append(images_e)
    for f in os.listdir(dir):
        ext = os.path.splitext(f)[1]
        if ext in extensions:
            file = os.path.join(dir,f)
            img = cv2.imread(file)
            image_e = ET.Element('image')
            image_e.set('file', str(file))
            e = (dlib.rectangle(left=1, top=1, right=img.shape[1]-1, bottom=img.shape[0]-1))
            shape = predictor(img, e)
            box = ET.Element('box')
            box.set('top', str(int(1)))
            box.set('left', str(int(1)))
            box.set('width', str(int(img.shape[1]-2)))
            box.set('height', str(int(img.shape[0]-2)))
            part_length = range(0,shape.num_parts) 
            for item, i in enumerate(sorted(part_length, key=str)):
                if ignore is not None:
                    if i not in ignore:
                        part = ET.Element('part')
                        part.set('name',str(int(i+1)))
                        part.set('x',str(int(shape.part(item).x)))
                        part.set('y',str(int(shape.part(item).y)))
                        box.append(part)
                else:
                    part = ET.Element('part')
                    part.set('name',str(int(i+1)))
                    part.set('x',str(int(shape.part(item).x)))
                    part.set('y',str(int(shape.part(item).y)))
                    box.append(part)
                
            box[:] = sorted(box, key=lambda child: (child.tag,float(child.get('name'))))
            image_e.append(box)
            images_e.append(image_e)
    et = ET.ElementTree(root)
    xmlstr = minidom.parseString(ET.tostring(et.getroot())).toprettyxml(indent="   ")
    with open(out_file, "w") as f:
        f.write(xmlstr)

def natural_sort_XY(l): 
    '''
    Internal function used by the dlib_xml_to_pandas. Performs the natural sorting of an array of XY 
    coordinate names.
    
    Parameters:
        l(array)=array to be sorted
        
    Returns:
        l(array): naturally sorted array
    '''
    convert = lambda text: int(text) if text.isdigit() else 0 
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key) ] 
    return sorted(l, key = alphanum_key)

def dlib_xml_to_pandas(xml_file: str, print_csv = False):
    '''
    Imports dlib xml data into a pandas dataframe. An optional file parsing argument is present
    for very specific applications. For most people, the parsing argument should remain as 'False'.
    
    Parameters:
        xml_file(str):file to be imported (dlib xml format)
        
    Returns:
        df(dataframe): returns a pandas dataframe containing the data in the xml_file. 
    '''
    tree=ET.parse(xml_file)
    root=tree.getroot()
    landmark_list=[]
    for images in root:
        for image in images:
            for boxes in image:
                box=boxes.attrib['top']\
                +'_'+boxes.attrib['left']\
                +'_'+boxes.attrib['width']\
                +'_'+boxes.attrib['height']
                for parts in boxes:
                    if parts.attrib['name'] is not None:
                        data={'id':image.attrib['file'],
                                'box_id':box,
                                'box_top':float(boxes.attrib['top']),
                                'box_left':float(boxes.attrib['left']),
                                'box_width':float(boxes.attrib['width']),
                                'box_height':float(boxes.attrib['height']),
                                'X'+parts.attrib['name']:float(parts.attrib['x']),
                                'Y'+parts.attrib['name']:float(parts.attrib['y']) }

                    landmark_list.append(data)
    dataset=pd.DataFrame(landmark_list)
    df = dataset.groupby(['id', 'box_id'], sort=False).max()
    df=df[natural_sort_XY(df)]
    if print_csv:
        basename = os.path.splitext(xml_file)[0]
        df.to_csv(f'{basename}.csv')
    return df


