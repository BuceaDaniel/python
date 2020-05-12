import tkinter as GUI
from PIL import Image, ImageTk
import matplotlib
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np
import cv2
from collections import Counter
from skimage.color import rgb2lab, deltaE_cie76
import os


def RGB2HEX(color):
    return "#{:02x}{:02x}{:02x}".format(int(color[0]), int(color[1]), int(color[2]))


def get_image(image_path):
    img = cv2.imread(image_path)

    # Extract Hue channel and make Numpy array for fast processing
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    mask = cv2.inRange(hsv, (35, 25, 25), (86, 255, 255))
    imask = mask > 0
    green = np.zeros_like(img, np.uint8)
    green[imask] = img[imask]
    print(mask.mean()*100/255)
    # Set all green pixels to 1

    # Now print percentage of green pixels
    cv2.imwrite("green.png", green)

    return img


def get_colors(image, number_of_colors):
    modified_image = cv2.resize(
        image, (600, 400), interpolation=cv2.INTER_AREA)

    modified_image = modified_image.reshape(
        modified_image.shape[0] * modified_image.shape[1], 3)

    clf = KMeans(n_clusters=number_of_colors)
    labels = clf.fit_predict(modified_image)

    counts = Counter(labels)

    # sort to ensure correct color percentage
    counts = dict(sorted(counts.items()))

    center_colors = clf.cluster_centers_
    # We get ordered colors by iterating through the keys
    ordered_colors = [center_colors[i] for i in counts.keys()]
    hex_colors = [RGB2HEX(ordered_colors[i]) for i in counts.keys()]
    rgb_colors = [ordered_colors[i] for i in counts.keys()]

    # if (show_chart):
    plt.figure(figsize=(8, 6))
    plt.pie(counts.values(), labels=hex_colors, colors=hex_colors)
    plt.show()

    return rgb_colors


def iniImage(filename):
    img = Image.open(filename)
    canvas.image = ImageTk.PhotoImage(img)
    canvas.create_image(0, 0, image=canvas.image, anchor="nw")


def browsefunc():
    global folder_path
    filename = GUI.filedialog.askopenfilename(filetypes=(
        ("JPG", "*.jpg"), ("PNG", "*.png"), ("SVG", "*.svg"), ("JPEG", "*.jpeg"),))
    folder_path = filename
    iniImage(folder_path)
    # get_colors(get_image(folder_path), 1)
    get_image(folder_path)


root = GUI.Tk()
folder_path = ""
canvas = GUI.Canvas(root, height=500, width=500, bg="grey")
canvas.pack(expand=True, fill="both")

browsebutton = GUI.Button(root, text="Browse Image", command=browsefunc)
browsebutton.pack()
root.mainloop()
# initGUI()
