"""
# DVA 2022: Exercise 1: Interactive Data Visualization in Python with Bokeh and Pandas

## Summary:
In this exercise, the ultimate goal is to get familiar with Bokeh and Pandas.

We will achieve this goal, by implementing a small "Data Inspector" tool. It allows us to select
two columns of our dataset, and plots them onto the x and y axis of a dot plot using Bokeh.

Essentially your task is to read through the code in dva_ex_1_firstname_lastname.py
and fill out the TODO code blocks according to the respective task.

Have a look at the demonstration video to have a reference on how the final tool should look and behave.

TIP 1:
if you haven't done the Bokeh Tutorial, I highly recommend you do so,
(https://mybinder.org/v2/gh/bokeh/bokeh-notebooks/master?filepath=tutorial%2F00%20-%20Introduction%20and%20Setup.ipynb)
or have a look at the "Server App" applications in the Bokeh gallery https://docs.bokeh.org/en/latest/docs/gallery.html

TIP 2:
Whenever you don't know something try google "Bokeh <your question / term>" or check the Bokeh docs or
check de discourse https://discourse.bokeh.org/


## Goals
**Goal 1**: Get familiar with the basics of Pandas

You should be able to:
- Load datasets from csv using pandas
- Combining datasets based on columns
- Renaming columns
- Basic cleaning (aka dropping nans)

**Goal 2**: Learn the Bokeh Server basics:

You should be able to:
- Run a Bokeh server application with the `bokeh serve` command
- Update the content a ColumnDataSource using dictionaries
- Add interactions with Bokeh widgets and implement callback functions


## How to run the code
Tip: If you have never used python before, or the following points could
just as well be part of the intergalactic highway scheme of Prostetnic Vogon Jeltz, please
Google "how to install requirements.txt in python tutorial" and
pick the one most attractive to you.


1. If you haven't install Python https://www.python.org/downloads/
   or Conda https://anaconda.org/ (conda is up to your )

2. Open a Terminal on your machine and move to the directory of the code:

    `cd /directory/of/this/code`
3. (Optional, but an indicator for good taste) Create a new virtual environment

    `python -m venv venv`

    Activate:

    Windows: `venv\Scripts\activate.bat`

    Else: `source venv/bin/activate`


4. Install the dependencies if you haven't before:

    `pip install -r requirements.txt`

5. Now you can always start your application directly with:
   `bokeh serve --show dva_ex_1_firstname_lastname.py`

6. Abort it with Ctrl + C

## Questions?
1. Google it
2. Ask a friend
3. Post to the forum
4. Ask during the office hour
5. (Last resort) Write me a mail halter@ifi.uzh.ch

"""
import typing

import numpy as np
import pandas as pd
from bokeh.layouts import layout
from bokeh.models import ColumnDataSource, Select, CustomJS
from bokeh.plotting import figure, curdoc, show
from bokeh.models.tools import HoverTool

'''
######################################################################
IMPORTANT: RENAME THIS FILE to dva_ex_1_firstname_lastname.py

           The code will be tested with Bokeh==2.4.2
           so make you have the same version or install
           the requirements.txt!
######################################################################


######################################
(0.5 Points) Section 1: Preprocessing
######################################
In this first section, it is your job to prepare the datasets to be visualized.
'''

# Load the datasets
df_sleep = pd.read_csv("mammals_sleep.csv")
df_predation = pd.read_csv("mammals_predation.csv")

# TIPP: Before advancing any further, inspect the datasets, how are they structured, what are the columns?
#       You can do this by using a simple print(df_sleep) or df_sleep.info()
#       Real-world datasets typically contain mistakes, typos, NaNs and other problems we have to deal with
#       Try to spot them early!

# print(df_sleep.dropna(axis="rows"))
# print(df_predation.dropna(axis="rows"))
# (0.1) In both datasets, drop all rows where the species in nan                                (please keep this line)
# TODO Here comes your code
df_sleep = df_sleep.dropna(axis="rows")
df_predation = df_predation.dropna(axis="rows")

# (0.1) Join the datasets based on the species column to one dataset                            (please keep this line)
# TODO Here comes your code
df_combined = df_sleep.set_index("species").join(df_predation.set_index("Species"))
##print(df_combined)
# (0.1) Remove all species where the body_wt is larger than 1000kg from the combined dataset    (please keep this line)
df_combined = df_combined[df_combined["body_wt"]<1000]

# (0.2) Rename all columns such that they do not contain any                                    (please keep this line)
# whitespaces and uppercase letters anymore
# eg. "Peter pan" -> "peter_pan"
# TODO Here comes your code
for i in list(df_combined):
    df_combined.rename(columns={i: i.lower()}, inplace=True)
    if " " in i:
        df_combined.rename(columns={i: i.replace(" ", "_")}, inplace=True)

"""
######################################
(1.5 Points) Section 2: Visualization
######################################

In this section, we will implement the actual visualization in Bokeh
The concept is as follows:

1. We have a ColumnDataSource "source" which holds our current data 
   (x-coordinates (xs), y-coordinates (ys) and the names of the species (species))
2. We have a Plot which displays the above ColumnDataSource using a circle glyph 
3. We have two Dropdown menues (called Select in Bokeh), where the user can select which column is used for the x-axis 
   or the y-axis respectively. 
4. Whenever the Selects change, we have to update the ColumnDataSource, this is done via a callback function

We begin with implementing a fetch_data function, which takes two Column names from the input table, 
and returns a new dataset with the corresponding values as x and y coordinates. 
"""


# (0.5 Points) Implement the fetch_data function according to it's docstrings                   (please keep this line)
def fetch_data(x_column_name: str, y_column_name: str):
    """
    (0.3 Points) Given two column names, this function returns a dictionary with
    - (xs): a list of x-coordinates
    - (ys): a list of y-coordinates
    - (species): a list of species names

    (0.2 Points) Ensures, the result does not contain any NaNs
    """
    # TODO Here comes your code
    xs = df_combined[x_column_name].values.tolist()
    ys = df_combined[y_column_name].values.tolist()
    species = df_combined.index.values.tolist()

    # Check if the values have a NaN or are empty or datavalue does not fit
    xs = [x for x in xs if pd.notnull(x) and x != ""]
    ys = [y for y in ys if pd.notnull(y) and y != ""]
    return dict(xs=xs, ys=ys, species=species)


# (0.2 Points) Create a ColumnDataSource with the data from fetch_data()                        (please keep this line)
# You can use any columns you want as the initial values

# TODO Here comes your code
x_val, y_val = "body_wt", "brain_wt"
source = ColumnDataSource(data=fetch_data(x_val, y_val))

# (0.2 Points) Create a figure with log axes, set initial axis labels correctly based            (please keep this line)
# on the previous step
# (0.1 Points) add tooltips with the species names, x and y coordinates                          (please keep this line)
# TODO Here comes your code
p = figure(
    x_axis_label=x_val,
    y_axis_label=y_val
)

hover = HoverTool()
hover.tooltips = [
    ("Species", "@species"),
    ("x", "@xs"),
    ("y", "@ys")
]
p.add_tools(hover)

# (0.1 Points) Create a circle glyph and bind it to the ColumnDataSource created previously      (please keep this line)
# TODO Here comes your code
p.circle(x='xs', y='ys', source=source)


def callback(attr, old, new):
    """
    This function is called whenever the current value of the select_xaxis or select_yaxis changes.

    Here, we have to
    - (0.1 Points) update the ColumnDataSource.data with a new dictionary returned from fetch_data
    - (0.1 Points) update the axis labels according to the new columns selected.

    Tipp:   Bokeh callbacks typically have this attr, old, new signature, however, in this case you can ignore
            them and fetch the current value of the two Select menus directly by accessing doing <your_select>.value

    """
    # TODO Here comes your code
    source.data = fetch_data(select_xaxis.value, select_yaxis.value)
    p.xaxis.axis_label = select_xaxis.value
    p.yaxis.axis_label = select_yaxis.value


# (0.3) Implement two Select Widgets and connect them to the callbacks                          (please keep this line)
#       Remove "species" from the list
# Tipp: If you are unsure on how to do this, have a look at the Callbacks Section and the Select Widget here:
# https://docs.bokeh.org/en/latest/docs/user_guide/interaction/widgets.html

# TODO Here comes your code
select_xaxis = Select(title="X - Axis", value=x_val, options=list(df_combined))
select_yaxis = Select(title="Y - Axis", value=y_val, options=list(df_combined))
select_xaxis.on_change("value", callback)
select_yaxis.on_change("value", callback)
# (0.1 Point) Add everything to the layout                                                      (please keep this line)
lt = layout(
    select_xaxis,
    select_yaxis,
    p,
)

# needs to be out commented
curdoc().add_root(lt)
curdoc().title = 'dva_ex1'
