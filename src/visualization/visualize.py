#import plotly.graph_objs as go
#import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

def visualize_trainingsdata(dataset):
    #fig = px.histogram(, x='language', template='plotly_white', title='Anzahl der Sprachen')
    #fig.update_xaxes(categoryorder='total descending').update_yaxes(title='Anzahl an Captions')
    #fig.show()
    print(dataset.describe())
    print(dataset.head())
    pass


def visualize_rankingdata():
    pass

def visualize_training():
    pass
    #try
    #ecept


def get_data(name):
    dataset = pd.read_csv('./masterarbeit_zur_wikipedia_image/data/' + name)
    return dataset


def start_visualization():
    print("Start Visualization...")
    wikidata = get_data('wikidata.csv')
    visualize_trainingsdata(wikidata)


start_visualization()