#!/usr/bin/env python
# coding: utf-8

# In[1]:


#DASHBOARD
import plotly.express as px
import plotly.graph_objs as go
import dash 
from dash import dcc, ctx
import dash_bootstrap_components as dbc
from dash import html
from jupyter_dash import JupyterDash
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate


import warnings
warnings.filterwarnings("ignore")


# In[2]:


#PROCESS
import geopandas as gpd
import pandas as pd
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import mapbox
from shapely.geometry import Point, Polygon, MultiPoint, GeometryCollection
from sklearn.cluster import DBSCAN
from alphashape import alphashape
import os


# In[3]:


#GOOGLE DRIVE API
from pydrive2.auth import GoogleAuth
from pydrive2.drive import GoogleDrive


# In[ ]:





# In[4]:


import io


# In[5]:


import requests


# In[6]:


import json


# In[7]:


# Set the API endpoint URL for JSON data.
url = 'https://api.jsonbin.io/v3/b/644b5b2eb89b1e2299931445/latest'


# In[8]:


# Set the headers for request
headers = {
    'X-Master-Key': '$2b$10$P8FyaIUtq5CHqVC/arfJf.nI0UyfBgTtT5t/RJtP.q./IKgXLEp/e',
    'Content-Type': 'application/json'
}


# In[9]:


# Send a GET request to the API endpoint to retrieve the JSON data
response = requests.get(url, headers=headers)


# In[10]:


# Parse the JSON data from the response
json_data = response.json()


# In[11]:


fish = json_data["record"]


# In[12]:


fish_str = json.dumps(fish)


# In[13]:


creds_json = json.loads(fish_str)


# In[14]:


def login_with_service_account():
    """
    Google Drive service with a service account.
    note: for the service account to work, you need to share the folder or
    files with the service account email.

    :return: google auth
    """
    # Define the settings dict to use a service account
    # We also can use all options available for the settings dict like
    # oauth_scope,save_credentials,etc.
    settings = {
    "client_config_backend": "service",
    "service_config": {
        "client_json_dict": creds_json
    }
    }
    # Create instance of GoogleAuth
    gauth = GoogleAuth(settings = settings)
    
    # Authenticate
    gauth.ServiceAuth()
    return gauth


# In[15]:


drive = GoogleDrive(login_with_service_account())


# In[16]:


#Access final_data folder using GOOGLE DRIVE API
#final data
drive = GoogleDrive(login_with_service_account())
final_data = drive.ListFile({'q': "'1txcW3LCzYPCJ_43KzU2wigWE1KwsqQ2s' in parents and trashed=false"}).GetList()
for data in final_data:
    print('title: %s, id: %s' % (data['title'], data['id']))


# In[17]:


#SORT FUNCTION
def sort_by_numeric_value(file):
    title = file['title']
    numeric_value = ''
    for char in title:
        if char.isdigit():
            numeric_value += char
        else:
            break
    return int(numeric_value) if numeric_value else 0

# Sort files based on numeric values in their titles
final_data = sorted(final_data, key=sort_by_numeric_value)


# In[18]:


for data in final_data:
    print('title: %s, id: %s' % (data['title'], data['id']))


# In[19]:


#sample access
final_data[0]['id']


# In[20]:


# Set the ID of the CSV file to access
#file_id = '11GkrsDHSjCKX2q6VV93l01AjNKWIipqH'

# Get the PyDrive file object
file_obj = drive.CreateFile({'id': final_data[0]['id']})

# Get the content of the file as a string
file_content = file_obj.GetContentString()

datfr = pd.read_csv(io.StringIO(file_content))


# In[21]:


app = JupyterDash(external_stylesheets=[dbc.themes.DARKLY])
server = app.server

mapbox_token = 'pk.eyJ1IjoiYmVybW9kYTA0IiwiYSI6ImNsZjBnbGxodjAxeHgzcm81eTRlazF5eDEifQ.0CPIhwqhMinleOCQ4sRHlQ'
px.set_mapbox_access_token(mapbox_token)


# In[22]:


#DATASET

# Get the PyDrive file object
file_obj = drive.CreateFile({'id': final_data[0]['id']})

# Get the content of the file as a string
file_content = file_obj.GetContentString()

dummy_df = pd.read_csv(io.StringIO(file_content))

dummy_gdf = gpd.GeoDataFrame(
    dummy_df, 
    geometry=gpd.points_from_xy(dummy_df.lon, dummy_df.lat)
)


# In[23]:


#global variables
fig = None
cluster_gdf = None
bar_fig = None
all_gdf = None


# In[24]:


#MULTI-CRITERIA DECISION MAKING

#PFZ threshold
def sst_threshold(sst):
    if sst < 24 or sst > 31:
        return 1.0
    elif sst >= 28 and sst <= 31:
        return 2.0
    elif sst >= 24 and sst <= 28:
        return 3.0
    else:
        return 0
    
#CHL-a threshold
def chla_threshold(chl_a):
    if chl_a < .2:
        return 1.0
    elif chl_a >= .2:
        return 2.0
    elif chl_a >= .4:
        return 3.0
    else:
        return 0

#MCDS
def MCDS (df) : 
    tf_weights = {'low': 1, 'moderate': 2, 'high': 3}
    pfz_df = df.loc[:, ['lat', 'lon']]

    pfz_df['sst_thresh'] = df['mean_sst'].apply(sst_threshold)
    pfz_df['chla_thresh'] = df['mean_chla'].apply(chla_threshold)
    pfz_df['tf_thresh'] = df['thermal_mask'].map(tf_weights)

    #weights for the variable for PFZ criteria
    w_sst = 0.2  
    w_chla = 0.4  
    w_tf = 0.4

    pfz_df['PFZ_score'] = w_sst*pfz_df['sst_thresh'] + w_chla*pfz_df['chla_thresh'] + w_tf*pfz_df['tf_thresh']
    pfz_df['pfz'] = np.where(pfz_df['PFZ_score'] >= 2.4, 'High', np.where(pfz_df['PFZ_score'] >= 1.7, 'Moderate', 'Low'))

    return pfz_df['PFZ_score'], pfz_df['pfz']


# In[25]:


# Set the ID of the shapefile folder
folder_id = '1-AKrUCl5IgtzKvGWtTL8SrtAcrCUScAH'

drive = GoogleDrive(login_with_service_account())
# Get a list of all the files in the folder
shapefile = drive.ListFile({'q': "'1-AKrUCl5IgtzKvGWtTL8SrtAcrCUScAH' in parents and trashed=false"}).GetList()

# Create a directory to store the downloaded files
os.makedirs('shapefiles', exist_ok=True)

# Download all the files in the folder to the shapefiles directory
for file in shapefile:
    file.GetContentFile(os.path.join('shapefiles', file['title']))

# Read the shapefile using GeoPandas
shapefile_path = 'shapefiles/dgp_divided_v3.shp'
dg_shape = gpd.read_file(shapefile_path)

converted_polygon = dg_shape['geometry'].buffer(0.0).simplify(0.0001, preserve_topology=False)
converted_area = converted_polygon.area * (111319.9 ** 2)
#area_m2 = dg_shape.area
area_km2 = converted_area / 1000000
dg_shape['area_km2'] = area_km2

dg_shape_final = dg_shape[dg_shape['area_km2'] != 0]
dg_shape_final['color_representation'] = 'Sea'


# In[26]:


# #DAVAO GULF SECTORS
# dg_shape = gpd.read_file('D:/usep acads/THESIS_FINAL_DATA/DAVAO_GULF/backup/dgp_divided_v3.shp')

# converted_polygon = dg_shape['geometry'].buffer(0.0).simplify(0.0001, preserve_topology=False)
# converted_area = converted_polygon.area * (111319.9 ** 2)
# #area_m2 = dg_shape.area
# area_km2 = converted_area / 1000000
# dg_shape['area_km2'] = area_km2

# dg_shape_final = dg_shape[dg_shape['area_km2'] != 0]
# dg_shape_final['color_representation'] = 'Sea'


# In[27]:


cluster_group_area = dg_shape_final[['POLY_ID', 'area_km2']]
cluster_group_area['High_area_km2'] = 0
cluster_group_area['Moderate_area_km2'] = 0
cluster_group_area['Low_area_km2'] = 0

def new_cga():
    global cluster_group_area
    
    cluster_group_area = dg_shape_final[['POLY_ID', 'area_km2']]
    cluster_group_area['High_area_km2'] = 0
    cluster_group_area['Moderate_area_km2'] = 0
    cluster_group_area['Low_area_km2'] = 0


# In[28]:


# function that generates a scatter mapbox figure and adds a choropleth overlay
def map_figure(mdf, color_drop):
    
    global fig  # declaring fig as a global variable so it can be accessed outside of the function
    
    # initialize color and color mode variables
    color = None
    color_mode = None
    
    # if statement to determine which color scheme to use based on the user's selection
    if color_drop == 1:
        color = 'pfz'
        color_mode = {'High': 'greenyellow', 'Moderate': 'steelblue', 'Low': 'midnightblue'}
        fig = px.scatter_mapbox(mdf,  # plot the scatter mapbox figure
                                lon=mdf['lon'],
                                lat=mdf['lat'],
                                hover_name='pfz',  # add 'pfz' as hover text
                                hover_data=['PFZ_score', 'mean_sst', 'mean_chla', 'mean_tm'],  # additional hover data
                                color=color,  # set color based on pfz values
                                color_discrete_map=color_mode,  # map color values to specific colors
                                opacity=.8)
        
    elif color_drop == 2:  # second color scheme option
        color = 'mean_sst'
        color_mode = 'turbo'
        fig = px.scatter_mapbox(mdf, lon=mdf['lon'], lat=mdf['lat'], hover_name='pfz',
                                hover_data=['PFZ_score', 'mean_sst', 'mean_chla', 'mean_tm'],
                                color=color, color_continuous_scale=color_mode, opacity=.8)
        
    else:  # third color scheme option
        color = 'mean_chla'
        color_mode = 'Viridis'
        fig = px.scatter_mapbox(mdf, lon=mdf['lon'], lat=mdf['lat'], hover_name='pfz',
                                hover_data=['PFZ_score', 'mean_sst', 'mean_chla', 'mean_tm'],
                                color=color, color_continuous_scale=color_mode, opacity=.8)

    # create choropleth mapbox figure using the dg_shape_final data
    chor_fig = px.choropleth_mapbox(dg_shape_final, geojson=dg_shape_final.geometry, color='color_representation',
                                    color_discrete_map={'Sea': 'midnightblue'}, locations=dg_shape_final.index,
                                    opacity=.7)

    # update the layout of the scatter mapbox figure
    fig.update_layout(
        mapbox=dict(
            center=dict(lat=6.747123, lon=125.7020),  # set the center of the map
            zoom=9,  # set the initial zoom level
            style="mapbox://styles/mapbox/dark-v10"),  # set the style of the map
        margin=dict(l=0, r=0, t=0, b=0),  # set the margin
        legend=dict(  # create legend
            bgcolor='rgba(255, 255, 255, 0.7)',
            x=0.05,
            y=0.05,
            traceorder='normal',
            font=dict(size=12),
            title_font=dict(size=14),
            orientation='h',
            yanchor='bottom',
            xanchor='left',
            bordercolor='#FFFFFF',
            borderwidth=1,
            itemsizing='trace',
            title=dict(text='Legend', side='top')
        )
    )

    # add the choropleth figure to the scatter mapbox figure
    fig.add_trace(chor_fig.data[0])


# In[29]:


drive = GoogleDrive(login_with_service_account())
poly_points = drive.ListFile({'q': "'1ap0HALyEGoTckALYdmeOvBft_BhEeU0z' in parents and trashed=false"}).GetList()

# Sort files based on numeric values in their titles
poly_points = sorted(poly_points, key=sort_by_numeric_value)


# In[30]:


sector_points_dict = {}



for index in range(len(poly_points)):
    
    # Get the PyDrive file object
    file_obj = drive.CreateFile({'id': poly_points[index]['id']})

    # Get the content of the file as a string
    file_content = file_obj.GetContentString()

    df = pd.read_csv(io.StringIO(file_content))
    
    gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.lon, df.lat))
    gdf.crs = 'WGS84'
    sector_points_dict[index] = gdf
    
    if index == 0:
        all_gdf = gdf
    else:
        all_gdf = pd.concat([all_gdf, gdf])



# In[31]:


def clustering(PFZ):
    
    # create an empty list to store information about each cluster
    cluster_polygons = []

    # loop through each key-value pair in the sector_points_dict dictionary
    for key, sector_points in sector_points_dict.items():

        # extract data for the current sector
        for_cluster = sector_points_dict[key]

        # check if any of the 'pfz' values for this sector contain the PFZ value passed to the function
        if for_cluster['pfz'].str.contains(str(PFZ)).any() :

            # filter the data for this sector to only include the PFZ value passed to the function
            for_cluster = for_cluster[for_cluster['pfz'] == str(PFZ)]

            # perform DBSCAN clustering on the lon/lat coordinates in the data
            dbscan = DBSCAN(eps= .006, min_samples= 5)
            for_cluster['cluster'] = dbscan.fit_predict(for_cluster[['lon', 'lat']])

            # remove any points that were not assigned to a cluster
            for_cluster = for_cluster[for_cluster['cluster'] != -1]

            # loop through each cluster identified by the algorithm
            for cluster_id in for_cluster['cluster'].unique():

                # extract the lon/lat coordinates for this cluster
                cluster_points = for_cluster[for_cluster['cluster'] == cluster_id]
                cluster_points = cluster_points[['lon', 'lat']].values

                # calculate the concave hull polygon for this cluster
                concave_hull = alphashape(cluster_points, alpha = 100)
                if  isinstance(concave_hull, Polygon):
                    cluster_polygon = Polygon(list(concave_hull.exterior.coords))

                    # add information about this cluster to the cluster_polygons list
                    cluster_polygons.append({'cluster_id': cluster_id, 'geometry': cluster_polygon, 'polygon_id': key})

    # convert the cluster_polygons list to a GeoDataFrame
    global cluster_gdf
    cluster_gdf = gpd.GeoDataFrame(cluster_polygons, crs=dummy_gdf.crs)

    # remove the 'cluster_id' column (which is redundant with the index)
    cluster_gdf.pop('cluster_id')

    # reassign the index to be the 'cluster_id' column
    clustered_index = cluster_gdf.index
    cluster_gdf['cluster_id'] = clustered_index

    # reorder the columns in the GeoDataFrame
    cluster_gdf = cluster_gdf[['cluster_id', 'geometry', 'polygon_id']]

    # calculate the area (in km^2) of each cluster polygon and add it as a new column
    converted_polygon = cluster_gdf['geometry'].buffer(0.0).simplify(0.0001, preserve_topology=False)
    converted_area = converted_polygon.area * (111319.9 ** 2)
    area_km2 = converted_area / 1000000
    cluster_gdf['area_km2'] = area_km2

    # add a new column to represent the color of the clusters in the resulting map
    cluster_gdf['color_representation'] = str(PFZ) + ' PFZ Clusters'


# In[32]:


cluster_gdf


# In[33]:


def cluster_trace(pfz):
    
    # global variables fig, cluster_gdf, and cluster_group_area
    global fig
    global cluster_gdf
    global cluster_group_area
    
    # Calls the function clustering and passes pfz as an argument
    clustering(pfz)
    color = None # Initializes variable color to None
    
    
    # Assigns a color based on the pfz parameter
    if pfz == "Low":
        color = "midnightblue"
    elif pfz == "Moderate":
        color = "steelblue"
    else:
        color = "greenyellow"
    
    # Creates a choropleth map
    chor_fig = px.choropleth_mapbox(cluster_gdf, 
                        geojson=cluster_gdf.geometry, 
                        color=cluster_gdf['color_representation'],
                        color_discrete_map={str(pfz) + ' PFZ Clusters': color},
                        locations=cluster_gdf.index, 
                        opacity=.5,)

    # Adds a trace to the fig plot. The trace is the first data element of chor_fig
    fig.add_trace(
        chor_fig.data[0]
    )

    # Groups the data in cluster_gdf by polygon_id and calculates the sum of area_km2. 
    cluster_group_area[pfz + "_area_km2"] = cluster_gdf.groupby('polygon_id')['area_km2'].sum()
    cluster_group_area = cluster_group_area.fillna(0) # Any missing values are filled with 0
    


# In[ ]:





# In[ ]:





# In[34]:


#MAPBOX TOKEN
mapbox_token = 'pk.eyJ1IjoiYmVybW9kYTA0IiwiYSI6ImNsZjBnbGxodjAxeHgzcm81eTRlazF5eDEifQ.0CPIhwqhMinleOCQ4sRHlQ'
px.set_mapbox_access_token(mapbox_token)


# In[35]:


# App layout
app.layout = dbc.Container([
    dbc.Row([
        dbc.Col([
            html.H1("fishDASH", style={'margin-left': '35px', 
                                                 'margin-top': '90px'}),
            html.P("FISHING ZONE MAP: A web application for finding the best fishing sites.", style={'margin-left': '35px'}),
            html.Label("Select a Month:", style={"font-size": "18px", 
                                                 'margin-left': '35px', 
                                                 'margin-top': '20px'}),
            dcc.Dropdown(id="months_drop",
                         options=[
                             {"label": "January", "value": 0},
                             {"label": "February", "value": 1},
                             {"label": "March", "value": 2},
                             {"label": "April", "value": 3},
                             {"label": "May", "value": 4},
                             {"label": "June", "value": 5},
                             {"label": "July", "value": 6},
                             {"label": "August", "value": 7},
                             {"label": "September", "value": 8},
                             {"label": "October", "value": 9},
                             {"label": "November", "value": 10},
                             {"label": "December", "value": 11}],
                         multi=False,
                         value=0,
                         style={'width': "90%",
                                'color':'black',
                                'margin-left': '20px', 
                                'margin-top': '18px',}
                         ),
            html.Label("Color Highlight:", style={"font-size": "18px",
                                                         'margin-left': '40px',
                                                         'margin-top': '30px'}),
            dcc.Dropdown(id = "color_drop",
                         options = [
                             {"label": "Potential Fishing Zone (PFZ)", "value" : 1},
                             {"label": "Sea Surface Temperature (SST)", "value" : 2},
                             {"label": "Sea Surface Chlorophyll-A Content (SSCC)", "value" : 3}],
                         multi = False,
                         value = 1,
                         style={'width': "90%",
                                'color':'black',
                                'margin-left': '20px', 
                                'margin-top': '15px'}
                        ),
            html.Div(
                dbc.Button('Generate', id='generate_button', n_clicks=0, color='info'),
                style={'text-align': 'center', 'margin-top': '15px'}
            ),
            html.Div([   
                html.H4("Sector Scope: ", style={'margin-top': '23px'}),
                html.P(["Sector 1: Don Marcelino", html.Br(),
                      "Sector 2: Don Marcelino(Outside Municipal Waters)", html.Br(),
                      "Sector 3: Malita", html.Br(),
                      "Sector 4: Governor Generoso", html.Br(),
                      "Sector 5: Sta. Maria, Sulop, Padada, Hagonoy, Digos", html.Br(),
                      "Sector 6: Outside Municipal Waters", html.Br(),
                      "Sector 7: Governor Generoso", html.Br(),
                      "Sector 8: Sta. Cruz, Davao City", html.Br(),
                      "Sector 9: San Isidro, Lupon, Banaybanay, Samal (Southern Region)", html.Br(),
                      "Sector 10: Davao City, Panabo, Tagum, Samal (Northern Region), Maco, Mabini, Pantukan"]
                      )
                ], style={'margin-left': '40px', 'max-width': '300px'})
                
        ], width=3, style={'outline': '1px solid white'}),
        
        dbc.Col([    
            dcc.Loading(
                html.Div(id='map-container', children=[
                    dcc.Store(id='store_data'),
                    dbc.Row([   
                        dbc.Col([
                            dcc.Graph(id='pfz_map')
                        ], width=12)
                    ]),
                    dbc.Row([
                        dbc.Col([
                            dcc.Graph(id='pfz_pie')
                        ], width=6),
                        dbc.Col([
                            dcc.Graph(id='area_bar')
                        ], width=6)
                    ])                
                ]),
                id="loading-2",
                type="graph"
            )
        ], width=9)
    ])
], fluid=True)


# In[ ]:





# In[36]:


# store callback test
@app.callback(
    [Output('store_data', 'data'),
     Output('pfz_map', 'figure'),
     Output('pfz_pie', 'figure'),
     Output('area_bar', 'figure')],
    [Input('generate_button', 'n_clicks')],
    [State('months_drop', 'value'),
     State('color_drop', 'value')]
)
def update_data_and_graphs(n_clicks, month, color):
    
    # Generate and return data to be stored
    my_data = {}
    stored_data = my_data
    
    colors = {
    'background': '#3e3d40',
    'text': 'white'
    }

    # Update graphs
    
    global fig
    global cluster_group_area
    global all_gdf
    global cluster_gdf
    
    if (month is None or month < 0) or (color is None or color < 0): 
        # Raises a PreventUpdate exception if month or color is not specified or is less than zero
        raise PreventUpdate
    else:

        bar_colors = ['steelblue', 'greenyellow', 'midnightblue'] # List of colors for each variable in the stacked bar
        # Get the PyDrive file object
        file_obj = drive.CreateFile({'id': final_data[month]['id']})

        # Get the content of the file as a string
        file_content = file_obj.GetContentString()

        month_df = pd.read_csv(io.StringIO(file_content))
        
        mdf = month_df
        
        mdf['PFZ_score'], mdf['pfz'] = MCDS(month_df) # Applies a function "MCDS" to the "month_df" DataFrame and adding two new columns: "PFZ_score" and "pfz"

        month_gdf = gpd.GeoDataFrame(
            mdf, 
            geometry=gpd.points_from_xy(mdf.lon, mdf.lat)
        )

        for key, sector_points in sector_points_dict.items():
            merged_df = pd.merge(sector_points_dict[key], month_gdf, on=['lat', 'lon'], how='inner', suffixes = ('_old', ''))
            merged_df = merged_df.filter(regex='^(?!.*_old)')
            sector_points_dict[key] = merged_df

        #update all_gdf
        update = pd.merge(all_gdf, month_gdf, on=['lat', 'lon'], how='inner', suffixes = ('_old', ''))
        update = update.filter(regex='^(?!.*_old)')
        all_gdf = update

        map_figure(mdf, color)
        new_cga()
        
        #PIE
        # Group the data by 'pfz' and count the occurrences of each group
        all_group = all_gdf.groupby('pfz').size().reset_index(name='count')
        
        # set dict of colors
        pie_color = {"Low": 'midnightblue', "Moderate": 'steelblue', "High": 'greenyellow'}

        # Create a pie chart with values as 'count' and names as 'pfz'
        pie_fig = px.pie(all_group, values='count', names='pfz', color = 'pfz',
                         color_discrete_map = pie_color, title="Percentage Distribution of Fishing Zones" )

        # Update the pie chart traces
        pie_fig.update_traces(
            # Set the colors of the chart
            marker=dict(
                #colors=bar_colors,
                # Set the color and width of the outline
                line=dict(color='white', width=1)
            ),
            # Set the font color and size for the text
            textfont=dict(
                color=colors['text'],
                size=14
            ),
            # Set the information that appears when hovering over the chart
            hoverinfo='label+percent+name',
            # Set the size of the hole in the center of the chart
            hole=0.5
        )

        # Update the layout of the chart
        pie_fig.update_layout(
            # Set the background color of the chart
            plot_bgcolor=colors['background'],
            # Set the background color of the paper
            paper_bgcolor=colors['background'],
            # Set the font color and size for the chart title and axis labels
            font=dict(color=colors['text'], size=14)
        )
        
        #map
        cluster_trace("Low")
        cluster_trace("Moderate")
        cluster_trace("High")

        # Creates a stacked bar chart showing fishing zone area for different sectors
        bar_fig = px.bar(cluster_group_area,
                         x='POLY_ID',
                         y=['Moderate_area_km2', 'High_area_km2', 'Low_area_km2'],
                         barmode='stack', title="Fishing Zone Area in Square Kilometers",
                        color_discrete_sequence=bar_colors)
        
        # Update the chart layout and formatting settings, including axis titles, background colors, and font styles
        bar_fig.update_layout(
            xaxis_title=dict(
                text="SECTORS",
                font=dict(color=colors['text'])
            ),
            yaxis_title=dict(
                text="area in km2",
                font=dict(color=colors['text'])
            ),
            plot_bgcolor=colors['background'],
            paper_bgcolor=colors['background'],
            font=dict(color=colors['text'], size=14)
        )
        
        # Update the traces of the chart (i.e., the bars)
        bar_fig.update_traces(
            marker=dict(
                  line=dict(color='#ffffff', width=1)
            ),
            textfont=dict(
                color=colors['text'],
                size=14
            )
        )
        
        # Update the x-axis and y-axis labels
        bar_fig.update_layout(xaxis_title="SECTORS", yaxis_title="area in km2")
         
    return stored_data, fig, pie_fig, bar_fig


# In[37]:


if __name__ == '__main__':
    app.run_server(debug = True, port = 8051)






