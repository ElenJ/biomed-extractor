#set PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python # run in command line before the app, while there are package incompabilities with python 3.12
import streamlit as st
import pandas as pd
from numpy.random import default_rng as rng
import numpy as np
import altair as alt

from datetime import time, datetime




st.title("My first app :-)")
st.header("This is a header with a divider", divider="rainbow")
st.subheader("Fingerübungen mit Streamlit")
user_value = st.slider("A fancy slider", min_value=0, max_value=100, value=25, step=5)
st.write("Your magic number is: ", user_value)
user_range = st.slider("Range slider", -10, 10, (-3, 3), step=1)
st.write("Your range is: ", user_range)
st.markdown("You can even handle ***time*** and :blue[**dates**]")
st.slider("A time slider", value=(time(11, 30), time(12, 45)))
user_date = st.slider("A date slider", value=datetime(2020, 1, 1, 9, 30), format="MM/DD/YY - hh:mm")
st.write("Your date is: ", user_date)
st.markdown("And for strings, you need to use ***select_slder***, though this seems stupid, needs a dropdown menu")
options=[
        "red",
        "orange",
        "yellow",
        "green",
        "blue",
        "indigo",
        "violet",
    ]
color = st.select_slider(
    "Select a color of the rainbow",
    options=options,
)
st.write("My favorite color is", color)
mycolor = st.selectbox("What is your favorite color?", options, index=3)
st.write("Your selected color is: ", mycolor)

st.markdown("For multiple selections, use **multiselect**")
mycolors = st.multiselect(
     'What are your favorite colors',
     options,
     ['yellow', 'red'])

st.write('You selected:', mycolors)

st.subheader("Plots")
mydf = pd.DataFrame(rng(0).standard_normal((100, 3)), columns=["a", "b", "c"])
st.line_chart(mydf[["a","b"]])

with st.sidebar:
    st.header("This is the sidebar")
    st.markdown("You can put :red[widgets] here")
    st.button("A button")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Just some text")
    st.write("Hello world")
    st.markdown("*Streamlit* is **really** ***cool***.")
    st.markdown('''
    :red[Streamlit] :orange[can] :green[write] :blue[text] :violet[in]
    :gray[pretty] :rainbow[colors] and :blue-background[highlight] text.''')


with col2:
    st.subheader("responsive graph")
    x = st.slider("Select a value", 0, 100, 25, step=5)
    st.write("Your selected value of :red[**x**] is", x)
    df = pd.DataFrame(rng(0).standard_normal((x,3)), columns=["a", "b", "c"])
    st.scatter_chart(df)

st.subheader("A dependent button")

if st.button("Press me"):
    st.write("You pressed the button!")
else:
    st.write("You haven't pressed the button yet.")

st.subheader("Writing examples")
st.write('Hello, *World!* :sunglasses:')
# Example 2

st.write(1234)

# Example 3

df = pd.DataFrame({
     'first column': [1, 2, 3, 4],
     'second column': [10, 20, 30, 40]
     })
st.write(df)

# Example 4

st.write('Below is a DataFrame:', df, 'Above is a dataframe.')

# Example 5

df2 = pd.DataFrame(
     np.random.randn(200, 3),
     columns=['a', 'b', 'c'])
c = alt.Chart(df2).mark_circle().encode(
     x='a', y='b', size='c', color='c', tooltip=['a', 'b', 'c'])
st.write(c)
