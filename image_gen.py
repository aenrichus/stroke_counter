from PIL import Image, ImageDraw, ImageFont
import os
import numpy as np
import pandas as pd


def gen_img(txt):
    width, height = 32, 32  # set the width and height of the image
    im = Image.new("F", (width, height), "#fff")  # create the 32-bit float image (with a white background)
    draw = ImageDraw.Draw(im)  # create an object that can be used to draw in the given image
    font = ImageFont.truetype("fonts/Inconsolata-Regular.ttf", 24)  # set the font and its size
    w, h = draw.textsize(txt, font=font)  # get the size of the text
    draw.text(((width-w)/2, (height-h)/2), txt, font=font, fill="#000")  # draw the text (in black, centered)
    path = 'images/'  # set the output location
    filename = txt + ".tiff"  # append the file type to the filename
    filename = os.path.join(path, filename)  # join the location and filename for saving
    im.save(filename)  # save the image to a file


def gen_imgs(indict):
    for i in indict:
        gen_img(i)


def load_image(infilename):
    img = Image.open(infilename)
    img.load()
    data = np.asarray(img, dtype="float32")
    flat_data = data.ravel()
    flat_data = flat_data / 255  # scales to 0-1
    return flat_data


def load_data(datafile):
    fromcsv = pd.read_table(datafile, encoding="utf-16")  # this gets edited sometimes
    #fromcsv = pd.read_csv(datafile)
    csvmat = fromcsv.as_matrix()
    return dict(zip(csvmat[:,0].tolist(), csvmat[:,1].tolist()))


# generate the images
#gen_imgs(load_data('joyo.txt'))  # WORD CANNOT BE NA


# import a list of training words and their lexicality from csv to a dictionary
#testing_words = load_data('list01_test.csv')
#print(testing_words)


#d.keys(), d.values() to get the dictionary objects: d.items() for the whole thing


# read a word as an array (to test)
#test = load_image("images/blire.jpg")  # returns the array
#print(test)
#testing = Image.fromarray(test, "L")  # converts array to image (to test)
#testing.save("test.jpeg")
