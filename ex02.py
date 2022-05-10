import glob
import os
import random

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


from sklearn.manifold import TSNE
from umap import UMAP
from sklearn.decomposition import PCA

from bokeh.plotting import figure, curdoc, show, row, column
from bokeh.models import ColumnDataSource, Slider, ImageURL, WheelZoomTool, PanTool, ResetTool, Range1d
from bokeh.layouts import layout

# Dependencies
# Make sure to install the additional dependencies noted in the requirements.txt using the following command:
# pip install -r requirements.txt

# You might want to implement a helper function for the update function below or you can do all the calculations in the
# update callback function.

####### HOW TO RUN ################
# Run EXACTLY this in your terminal
# no, really, exactly this ;-) otherwise the images don't show up
#
# python -m bokeh serve --show .
###################################


#############################
# Preprocessing #     1 Point
#############################


# Fetch the number of images using glob or some other path analyzer
N = len(glob.glob("static/*.jpg"))

# Find the root directory of your app to generate the image URL for the bokeh server
ROOT = os.path.split(os.path.abspath(os.path.dirname(__file__)))[1] + "/"

# Number of bins per color for the 3D color histograms
N_BINS_COLOR = 15

# Define an array containing the 3D color histograms. We have one histogram per image each having N_BINS_COLOR^3 bins.
# i.e. an N * N_BINS_COLOR^3 array
colorHistogram = []


# initialize an empty list for the image file paths
imageFilePath = []

# Compute the color and channel histograms
for idx, f in enumerate(glob.glob("static/*.jpg")):
    # open image using PILs Image package
    img = Image.open(f)

    # Convert the image into a numpy array and reshape it such that we have an array with the dimensions (N_Pixel, 3)
    a = np.asarray(img).reshape((img.width*img.height, 3))

    # Compute a multi dimensional histogram for the pixels, which returns a cube
    # reference: https://numpy.org/doc/stable/reference/generated/numpy.histogramdd.html
    histogram, edges = np.histogramdd(a, bins=(N_BINS_COLOR, N_BINS_COLOR, N_BINS_COLOR))
    # However, later used methods do not accept multi dimensional arrays, so reshape it to only have columns and rows
    # (N_Images, N_BINS^3) and add it to the color_histograms array you defined earlier
    # reference: https://numpy.org/doc/stable/reference/generated/numpy.reshape.html
    p = np.reshape(histogram, (1, N_BINS_COLOR**3))
    colorHistogram.append(p)
    # Append the image url to the list for the server
    url = ROOT + f
    imageFilePath.append(url)
colorHistogram = np.reshape(colorHistogram, (N, N_BINS_COLOR**3))


def compute_pca(n_components=2) -> np.ndarray:
    pca = PCA(n_components=n_components)
    pca.fit(colorHistogram)
    sol = pca.transform(colorHistogram)
    return sol

def compute_umap(n_neighbors=15) -> np.ndarray:
    """performes a UMAP dimensional reduction on color_histograms using given n_neighbors"""
    # compute and return the new UMAP dimensional reduction
    reducer = UMAP(n_neighbors=n_neighbors, n_components=2)
    reducer.fit(colorHistogram)
    embedding = reducer.transform(colorHistogram)
    # Verify that the result of calling transform is
    # idenitical to accessing the embedding_ attribute
    assert (np.all(embedding == reducer.embedding_))
    return embedding

def on_update_umap(old, attr, new):
    """callback which computes the new UMAP mapping and updates the source_umap"""
    # Compute the new t-sne using compute_umap

    # update the source_umap

    source_umap.data = dict(
        url3=imageFilePath,
        x3=[i[0] for i in compute_umap(umap_n_neigbors.value_throttled)],
        y3=[i[0] for i in compute_umap(umap_n_neigbors.value_throttled)],
        w3=[20] * 121,
        h3=[10] * 121)


def compute_tsne(perplexity=4, early_exaggeration=10) -> np.ndarray:
    """performes a t-SNE dimensional reduction on color_histograms using given perplexity and early_exaggeration"""
    # compute and return the new t-SNE dimensional reduction
    X_embedded = TSNE(n_components=2, learning_rate='auto',
                      perplexity=perplexity,
                      early_exaggeration=early_exaggeration).fit_transform(colorHistogram)
    return X_embedded
    # return ...


def on_update_tsne(old, attr, new):
    """callback which computes the new t-SNE mapping and updates the source_tsne"""
    # Compute the new t-sne using compute_tsne
    source_tsne.data = dict(
        url2=imageFilePath,
        x2=[i[0] for i in compute_tsne(tsne_perplexity.value_throttled, tsne_early_exaggeration.value_throttled)],
        y2=[i[0] for i in compute_tsne(tsne_perplexity.value_throttled, tsne_early_exaggeration.value_throttled)],
        w2=[12] * 121,
        h2=[8] * 121)

    # update the source_tsne

########################################
# Section ColumnDataSources # 0.5 Points
########################################

# Calculate the indicated dimensionality reductions
# references:
# https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html
# https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html
# https://umap-learn.readthedocs.io/en/latest/basic_usage.html

# Calculate the indicated dimensionality reductions


# Construct three data sources, one for each dimensional reduction,
# each containing the respective dimensional reduction result and the image paths


source_pca = ColumnDataSource(data=dict(
    url=imageFilePath,
    x1=[i[0] for i in compute_pca()],
    y1=[i[1] for i in compute_pca()],
    w1=[40000]*121,
    h1=[20000]*121))
source_tsne = ColumnDataSource(data = dict(
    url2=imageFilePath,
    x2=[i[0] for i in compute_tsne()],
    y2=[i[0] for i in compute_tsne()],
    w2=[20]*121,
    h2=[10]*121))
source_umap = ColumnDataSource(data = dict(
    url3=imageFilePath,
    x3=[i[0] for i in compute_umap()],
    y3=[i[0] for i in compute_umap()],
    w3=[12]*121,
    h3=[8]*121))




#############################
# Section Plots #  0.5 Points
#############################

# Create a first figure for the PCA data. Add the wheel_zoom, pan and reset tools to it.

# And use bokehs image_url to plot the images as glyphs
# reference: https://docs.bokeh.org/en/latest/docs/reference/models/glyphs/image_url.html

# Create a second plot for the t-SNE result in the same fashion as the previous.

# Create a third plot for the UMAP result in the same fashion as the previous.

# Create a slider to control the t-SNE hyperparameter "perplexity" with a range from 2 to 20 and a title "Perplexity"


#################################
# Section Callbacks #  1.0 Points
#################################

# Create a callback, such that whenever the value of the slider changes, on_update_tsne is called.
# Tipp: Using "value_throttled" instead of "value" ensures the callback is only fired when the mouse has stopped moving
# this helps reducing computation when the callback is expensive to compute
# ref https://stackoverflow.com/questions/38375961/throttling-in-bokeh-application

# Create a second slider to control the t-SNE hyperparameter "early_exaggeration"
# with a range from 2 to 50 and a title "Perplexity"

# Connect it to the on_update_tsne callback in the same fashion as the previous slider


# Create a third slider to control the UMAP hyperparameter "n_neighbors"

# Connect it to the on_update_umap callback in the same fashion as the previous slider

# You can use the command below in the folder of your python file to start a bokeh directory app.
# Be aware that your python file must be named main.py and that your images have to be in a subfolder name "static"

# bokeh serve --show .
# python -m bokeh serve --show .

# dev option:
# bokeh serve --dev --show .
# python -m bokeh serve --dev --show .
from bokeh.models import Grid, ImageURL, LinearAxis, Plot


plot = Plot(title="PCA", width=400, height=400)

image1 = ImageURL(url="url", x="x1", y="y1", w="w1", h="h1", anchor="center")
plot.add_glyph(source_pca, image1)
plot.add_tools(WheelZoomTool(), PanTool(), ResetTool())


xaxis = LinearAxis()
plot.add_layout(xaxis, 'below')

yaxis = LinearAxis()
plot.add_layout(yaxis,'left')

plot.add_layout(Grid(dimension=0, ticker=xaxis.ticker))
plot.add_layout(Grid(dimension=1, ticker=yaxis.ticker))

tsne_perplexity = Slider(start=2, end=20, step=1, value=10, title="Perplexity")
tsne_early_exaggeration = Slider(start=2, end=50, step=1, value=4, title="early_exaggeration")
tsne_perplexity.on_change("value_throttled", on_update_tsne)
tsne_early_exaggeration.on_change("value_throttled", on_update_tsne)
plot2 = Plot(title="TSNE", width=400, height=400,x_range = Range1d(start=-50, end=50),
y_range=Range1d(start=-50, end=50))
plot2.add_tools(WheelZoomTool(), PanTool(), ResetTool())

image2 = ImageURL(url="url2", x="x2", y="y2", w="w2", h="h2", anchor="center")
plot2.add_glyph(source_tsne, image2)


xaxis2 = LinearAxis()
plot2.add_layout(xaxis2, 'below')

yaxis2 = LinearAxis()
plot2.add_layout(yaxis2,'left')

plot2.add_layout(Grid(dimension=0, ticker=xaxis2.ticker))
plot2.add_layout(Grid(dimension=1, ticker=yaxis2.ticker))

umap_n_neigbors = Slider(start=2, end=20, step=1, value=15, title="N_Neighbors")
umap_n_neigbors.on_change("value_throttled", on_update_umap)
plot3 = Plot(title="UMAP", width=400, height=400, x_range = Range1d(start=-20, end=40),
y_range=Range1d(start=-20, end=40))

image3 = ImageURL(url="url3", x="x3", y="y3", w="w3", h="h3", anchor="center")
plot3.add_glyph(source_umap, image3)
plot3.add_tools(WheelZoomTool(), PanTool(), ResetTool())

xaxis3 = LinearAxis()
plot3.add_layout(xaxis3, 'below')

yaxis3 = LinearAxis()
plot3.add_layout(yaxis3,'left')

plot3.add_layout(Grid(dimension=0, ticker=xaxis3.ticker))
plot3.add_layout(Grid(dimension=1, ticker=yaxis3.ticker))

curdoc().add_root(row(plot,column(plot2, tsne_perplexity, tsne_early_exaggeration),column(plot3, umap_n_neigbors)))
show(row(plot,column(plot2, tsne_perplexity, tsne_early_exaggeration),column(plot3, umap_n_neigbors)))