import numpy as np
import pandas as pd
import random

from typing import List, Tuple
from bokeh.models import ColumnDataSource, Slider, Div, Select
from bokeh.sampledata.iris import flowers
from bokeh.plotting import figure, curdoc
from bokeh.layouts import column, row
from bokeh.palettes import Spectral10
from bokeh.transform import factor_cmap

# Use these centroids in the first iteration of you algorithm if "Random Centroids" is set to False in the Dashboard
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


def get_closest(data_point: np.ndarray, centroids: np.ndarray):
    """
    Takes a data_point and a nd.array of multiple centroids and returns the index of the centroid closest to data_point
    by computing the euclidean distance for each centroid and picking the closest.
    """
    N = centroids.shape[0]
    dist = np.empty(N)
    for i, c in enumerate(centroids):
        dist[i] = np.linalg.norm(c - data_point)
    index_min = np.argmin(dist)
    return index_min


def k_means(data: np.ndarray, k: int = 3, n_iter: int = 500, random_initialization=False) -> Tuple[np.ndarray, int]:
    """
    :param data: your data, a numpy array with shape (n_entries, n_features)
    :param k: The number of clusters to compute
    :param n_iter: The maximal numnber of iterations
    :param random_initialization: If False, DEFAULT_CENTROIDS are used as the centroids of the first iteration.

    :return: A tuple (cluster_indices: A numpy array of cluster_indices,
                      n_iterations: the number of iterations it took until the algorithm terminated)
    """
    # Initialize the algorithm by assigning random cluster labels to each entry in your dataset
    k = k + 1
    centroids = data[random.sample(range(len(data)), k)]
    labels = np.array([np.argmin([(el - c) ** 2 for c in centroids]) for el in data])
    clustering = []
    for k in range(k):
        clustering.append(data[labels == k])

    # Implement K-Means with a while loop, which terminates either if the centroids don't move anymore, or
    # if the number of iterations exceeds n_iter
    counter = 0
    while counter < n_iter:
        # Compute the new centroids, if random_initialization is false use DEFAULT_CENTROIDS in the first iteration
        # if you use DEFAULT_CENTROIDS, make sure to only pick the k first entries from them.
        if random_initialization is False and counter == 0:
            centroids = DEFAULT_CENTROIDS[random.sample(range(len(DEFAULT_CENTROIDS)), k)]

        # Update the clusters
        labels = np.array([get_closest(el, centroids) for el in data])
        clustering = []
        for i in range(k):
            clustering.append(np.where(labels == i)[0])

        counter += 1

        new_centroids = np.zeros_like(centroids)
        for i in range(k):
            if len(clustering[i]) > 0:
                new_centroids[i] = data_np[clustering[i]].mean(axis=0)
            else:
                new_centroids[i] = centroids[i]

        # exit the while loop, if the centroids didn't move
        if clustering is not None and (centroids != new_centroids).sum() == 0:
            break
        else:
            centroids = new_centroids

    num_samples = sum(x.shape[0] for x in clustering)
    indices = np.empty((num_samples,))  # An empty array with correct size
    for i, c in enumerate(clustering):
        # use cluster indices to assign to correct the cluster index
        indices[c] = i
    return indices.astype(int), counter


def callback(attr, old, new):
    # recompute the clustering and update the colors of the data points based on the result
    k = slider.value_throttled
    init = dropdown.value
    new_cluster, counter_new = k_means(data_np, k, 500, init)

    source.data['clustering'] = new_cluster.astype(str)
    mapper = factor_cmap('clustering', palette=Spectral10, factors=np.unique(new_cluster).astype(str))
    scatter_plot1.glyph.fill_color = mapper
    scatter_plot1.glyph.line_color = mapper
    scatter_plot2.glyph.fill_color = mapper
    scatter_plot2.glyph.line_color = mapper
    div.text = 'Number of iterations: %d' % (counter_new)


# read and store the dataset
data: pd.DataFrame = flowers.copy(deep=True)
data = data.drop(['species'], axis=1)

# Create a copy of the data as numpy array, which you can use for computing the clustering
data_np = np.asarray(data)
# Create the dashboard
# 1. A Select widget to choose between random initialization or using the DEFAULT_CENTROIDS on top
dropdown = Select(title='Random Centroids', value='False', options=['True', 'False'])
# 2. A Slider to choose a k between 2 and 10 (k being the number of clusters)
slider = Slider(start=2, end=10, value=3, step=1, title='k')
# 4. Connect both widgets to the callback
dropdown.on_change('value', callback)
slider.on_change('value_throttled', callback)
# 3. A ColumnDataSource to hold the data and the color of each point you need
clustering, counter = k_means(data_np, 3, 500, False)
source = ColumnDataSource(dict(petal_length=data['petal_length'],
                               sepal_length=data['sepal_length'],
                               petal_width=data['petal_width'],
                               clustering=clustering.astype(str)))
# 4. Two plots displaying the dataset based on the following table, have a look at the images
# in the handout if this confuses you.
#
#       Axis/Plot   Plot1   Plot2
#       X   Petal length    Petal width
#       Y   Sepal length    Petal length
#
# Use a categorical color mapping, such as Spectral10, have a look at this section of the bokeh docs:
# https://docs.bokeh.org/en/latest/docs/user_guide/categorical.html#filling
mapper = factor_cmap('clustering',palette=Spectral10,factors=np.unique(clustering).astype(str))

p1 = figure(title='Scatterplot of flowers distribution by petal length and sepal length')
p1.yaxis.axis_label = 'Sepal length'
p1.xaxis.axis_label = 'Petal length'
scatter_plot1 = p1.scatter(x='petal_length',
                           y='sepal_length',
                           fill_alpha=0.4,
                           size=10,
                           source=source,
                           fill_color=mapper,
                           line_color=mapper)

p2 = figure(title='Scatterplot of flowers distribution by petal width and petal length')
p2.yaxis.axis_label = 'Petal length'
p2.xaxis.axis_label = 'Petal width'
scatter_plot2 = p2.scatter(x='petal_width',
                           y='petal_length',
                           fill_alpha=0.4,
                           source=source,
                           size=10,
                           fill_color=mapper,
                           line_color=mapper)
# 5. A Div displaying the currently number of iterations it took the algorithm to update the plot.
div = Div(text='Number of iterations: %d' % (counter))
div.on_change('text', callback)

lt = row(column(dropdown, slider, div), p1, p2)

curdoc().add_root(lt)