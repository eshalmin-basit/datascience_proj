import pandas as pd
import streamlit as st
import plotly.express as px
from PIL import Image
import csv
import plotly.graph_objects as go

st.title("The General Social Survery Data Analytics Web App")

st.write("  ")
st.write("  ")
st.write("  ")
st.header("GSS 2016 Dataset")
data=pd.read_csv(r"D:\datascience_projects\Data Anylatics APP\gss2016.csv",)
st.dataframe(data)
data.head()
st.write("  ")
st.write("  ")
st.write("  ")
st.header("GSS data filtered on the basis of race,sex,age,degree,wrkstat,income,happy")
st.write("  ")
st.write("  ")
data_fil=data[['race','sex','age','degree','wrkstat','income','happy']]
st.dataframe(data_fil)

st.markdown("This is the list of columns")
column={'race','sex','age','degree','wrkstat','income','happy'}
st.write("  ")
st.write("  ")
st.header("GSS data aggregated by count")

pick_cols=st.selectbox("count by columns:",list(column))
data_fil["Count"]=0
data_fil_count=data_fil.groupby(pick_cols).count()
data_fil_count=data_fil_count[["Count"]]
data_fil_count["Percentages"]=(data_fil_count.Count/data_fil_count.Count.sum())*100
st.dataframe(data_fil_count)
st.write("  ")
st.write("  ")
st.write("  ")


st.header("GSS Data correlation between columns")
multi_select_cols=st.multiselect("Multi-select columns for correlations:",list(column),default=['sex'])
multi_select_data_fil=data_fil[multi_select_cols]
st.dataframe(multi_select_data_fil)
st.write("  ")
st.write("  ")

st.header("GSS data correlation between multiple columns")
multi_select_cols2=st.multiselect("Multiple select columns group by:",list(column),default=['sex'])
multi_select_groupby=data_fil[multi_select_cols2].groupby(multi_select_cols2).size().reset_index(name="Count")
multi_select_groupby["Percentages"]=(multi_select_groupby.Count/multi_select_groupby.Count.sum())*100
st.dataframe(multi_select_groupby)
st.write("  ")
st.write("  ")
#visualizing the data
st.header("GSS DATA Aggregated by Count PIE CHART")
pick_cols_vis=st.selectbox("Visualize by column:",list(column))
data_fil_count_vis=data_fil.groupby(pick_cols_vis).count()
data_fil_count_vis['x-axis']=data_fil_count_vis.index
fig=go.Figure(data=[go.Pie(labels=data_fil_count_vis['x-axis'],values=data_fil_count_vis["Count"])])
st.plotly_chart(fig)
