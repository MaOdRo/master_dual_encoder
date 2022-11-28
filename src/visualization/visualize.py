#import plotly.graph_objs as go
#import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.graph_objs as go
import plotly.express as px
import pandas as pd

def visualize_trainingsdata(dataset):
    print(dataset.head())
    pass

def visualize_language(dataset):
    #Sprachverteilung im Datensatz
    fig = px.histogram(dataset, x='language', template='plotly_white', title='Anzahl der Sprachen')
    fig.update_xaxes(categoryorder='total descending').update_yaxes(title='Anzahl an Captions')
    fig.show()

def visualize_language_top5(dataset):
    dfg = dataset.groupby(['language']).size().to_frame().sort_values([0], ascending = False).head(5).reset_index()
    dfg.columns = ['language', 'count']
    fig = px.histogram(dfg, x='language', y = 'count')
    fig.layout.yaxis.title.text = 'count'
    fig.show()


def get_data(name):
    dataset = pd.read_csv('../master_dual_encoder/data/' + name)
    return dataset

def visualizing_data():    
    print("Header der Trainingsdaten:")
    print(" ")
    wikidata = get_data('wikidata.csv')
    visualize_trainingsdata(wikidata)
    print("#"*150)
    print(" ")
    print("Header der Testdaten...")
    print(" ")
    wikidata_test = get_data('wiki_test.csv')
    visualize_trainingsdata(wikidata_test)

    visualize_language(wikidata)
    visualize_language_top5(wikidata)

visualizing_data()

