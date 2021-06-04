import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib
from PIL import Image
img = Image.open("heart.jpeg")
st.image(img,width=750)
st.title("Heart Attack analysis and prediction dataset",)
df=pd.read_csv('heart.csv')
rows=df.head(25)
st.table(rows)
st.header("Visualization of Heart Attack analysis and prediction dataset.")
st.set_option('deprecation.showPyplotGlobalUse', False)
selection=st.selectbox("Select the plot model",['Barplot','pairplot','heatmap','jointplot','Displot'])
if selection=='Barplot':
    st.subheader("Bar Plot")
    df.plot.bar(x='age', y='trtbps')
    st.pyplot()
elif selection=='jointplot':
    st.subheader("Jointplot")
    sns.jointplot(x='age', y='chol', data=df, kind='scatter')
    st.pyplot()
elif selection == 'pairplot':
    st.subheader("Pairplot")
    sns.pairplot(df,hue='sex',palette="rainbow")
    st.pyplot()
elif selection == 'Displot':
    st.subheader("Displot")
    sns.displot(data=df, x="trtbps", hue="sex",kind="kde")
    st.pyplot()
else:
    st.subheader("Heatmap")
    sns.heatmap(df.corr(), cmap='coolwarm', annot=True)
    st.pyplot()



