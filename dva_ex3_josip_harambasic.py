import numpy as np
import pandas as pd

from typing import List, Tuple
from bokeh.plotting import figure, show
from bokeh.models import ColumnDataSource, Slider, Div, Select
from bokeh.sampledata.iris import flowers
from bokeh.plotting import figure, curdoc
from bokeh.layouts import column, row, layout
from bokeh.palettes import Spectral10
from bokeh.transform import factor_cmap
from bokeh.models import Grid, ImageURL, LinearAxis, Plot

# Use these centroids in the first iteration of your algorithm if "Random Centroids" is set to False in the Dashboard
DEFAULT_CENTROIDS = np.array([[5.664705882352942, 3.0352941176470587, 3.3352941176470585, 1.0176470588235293],
                              [5.446153846153847, 3.2538461538461543, 2.9538461538461536, 0.8846153846153846],
                              [5.906666666666667, 2.933333333333333, 4.1000000000000005, 1.3866666666666667],
                              [5.992307692307692, 3.0230769230769234, 4.076923076923077, 1.3461538461538463],
                              [5.747619047619048, 3.0714285714285716, 3.6238095238095243, 1.1380952380952383],
                              [6.161538461538462, 3.030769230769231, 4.484615384615385, 1.5307692307692309],
                              [6.294117647058823, 2.9764705882352938, 4.494117647058823, 1.4],
                              [5.853846153846154, 3.215384615384615, 3.730769230769231, 1.2076923076923078],
                              [5.52857142857143, 3.142857142857143, 3.107142857142857, 1.007142857142857],
                              [5.828571428571429, 2.9357142857142855, 3.664285714285714, 1.1]])


selectingTool = Select(title="Random Centroids", value="False", options=list(["True","False"]))
upper_slider = Slider(start=2, end=10, step=1, value=3, title="k")

colormap = {'setosa': Spectral10[0], 'versicolor': Spectral10[1], 'virginica': Spectral10[2]}
colors = [colormap[x] for x in flowers['species']]

p = figure(title="Scatterplot of flower distribution by petal length and sepal length", width=450, height=450)
p.xaxis.axis_label = 'Petal Length'
p.yaxis.axis_label = 'Sepal length'

p.scatter(flowers["petal_length"], flowers["sepal_length"],
          color=colors, fill_alpha=0.2, size=10)

#################################################################################

p2 = figure(title="Scatterplot of flower distribution by petal width and petal width",width=450, height=450)
p2.xaxis.axis_label = 'Petal width'
p2.yaxis.axis_label = 'Petal length'

p2.scatter(flowers["petal_width"], flowers["petal_length"],
          color=colors, fill_alpha=0.2, size=10)

#################################################################################
selectingTool2 = Select(title="Random Centroids", value="False", options=list(["True","False"]))
upper_slider2 = Slider(start=2, end=10, step=1, value=3, title="k")

colormap = {'setosa': Spectral10[0], 'versicolor': Spectral10[1], 'virginica': Spectral10[2]}
colors = [colormap[x] for x in flowers['species']]

p3 = figure(title="Scatterplot of flower distribution by petal length and sepal length", width=450, height=450)
p3.xaxis.axis_label = 'Petal Length'
p3.yaxis.axis_label = 'Sepal length'

p3.scatter(flowers["petal_length"], flowers["sepal_length"],
          color=colors, fill_alpha=0.2, size=10)

#################################################################################

p4 = figure(title="Scatterplot of flower distribution by petal width and petal width",width=450, height=450)
p4.xaxis.axis_label = 'Petal width'
p4.yaxis.axis_label = 'Petal length'

p4.scatter(flowers["petal_width"], flowers["petal_length"],
          color=colors, fill_alpha=0.2, size=10)

lt = layout(
    column(row(column(selectingTool,upper_slider),p,p2), row(column(selectingTool2,upper_slider2),p3,p4))
)

curdoc().add_root(lt)
curdoc().title = "DVA_ex_3"

#show(p)


def get_closest(data_point: np.ndarray, centroids: np.ndarray):
    """
    Takes a data_point and a nd.array of multiple centroids and returns the index of the centroid closest to data_point
    by computing the euclidean distance for each centroid and picking the closest.

    :param data_point:
    :param centroids:
    :return:
    """
    return # the index of the centroid closest to the datapoint


def k_means(data: np.ndarray, k:int=3, n_iter:int=500, random_initialization=False) -> Tuple[np.ndarray, int]:
    """
    Your k-means implementation. (of course, no other library can be used for this) of course checking the internet
    for other implementations is a good way to start!

    :param data: your data, a numpy array with shape (n_entries, n_features)
    :param k: The number of clusters to compute
    :param n_iter: The maximal numnber of iterations
    :param random_initialization: If False, DEFAULT_CENTROIDS are used as the centroids of the first iteration.

    :return: A tuple (cluster_indices: A numpy array of cluster_indices,
                      n_iterations: the number of iterations it took until the algorithm terminated)
    """
    # Initialize the algorithm by assigning random cluster labels to each entry in your dataset
    clustering = ...

    # Implement K-Means with a while loop, which terminates either if the centroids don't move anymore, or
    # if the number of iterations exceeds n_iter
    counter = 0
    while counter < n_iter:
        # Compute the new centroids, if random_initialization is false use DEFAULT_CENTROIDS in the first iteration
        # if you use DEFAULT_CENTROIDS, make sure to only pick the k first entries from them.


        # Update the cluster labels using get_closest


        # if the centroids didn't move, exit the while loop
        pass

    # return the final cluster labels and the number of iterations it took
    return clustering, counter


def callback(attr, old, new):
    # recompute the clustering and update the colors of the data points based on the result
    pass

# read and store the dataset
#data: pd.DataFrame = flowers.copy(deep=True)
#data = data.drop(['species'], axis=1)

# Create a copy of the data as numpy array, which you can use for computing the clustering

# Create the dashboard
# 1. A Select widget to choose between random initialization or using the DEFAULT_CENTROIDS on top
# 2. A Slider to choose a k between 2 and 10 (k being the number of clusters)
# 4. Connect both widgets to the callback
# 3. A ColumnDataSource to hold the data and the color of each point you need
# 4. Two plots displaying the dataset based on the following table, have a look at the images
# in the handout if this confuses you.
#
#       Axis/Plot	Plot1 	Plot2
#       X	Petal length 	Petal width
#       Y	Sepal length	Petal length
#
# Use a categorical color mapping, such as Spectral10, have a look at this section of the bokeh docs:
# https://docs.bokeh.org/en/latest/docs/user_guide/categorical.html#filling
# 5. A Div displaying the currently number of iterations it took the algorithm to update the plot.

#lt = row(...your plot layout)

#curdoc().add_root(lt)
#curdoc().title = "DVA_ex_3"


# Run it with
# bokeh serve --show dva_ex3_yourname.py