import dash
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd
import plotly.express as px
import wandb
import tensorflow as tf
import h5py
import numpy as np
import plotly.express as px
import flask
import scipy.stats

DATA_DIR='/home/shush/profile/QuantPred/datasets/ATAC_v2/grid5/pred_i_3072_w_128_bpnet_fftmse.h5'
CELL_LINE=1
server = flask.Flask(__name__)
app = dash.Dash(__name__, server=server)

def np_mse(a, b):
    return ((a - b)**2)

def scipy_pr(y_true, y_pred):

    pr = scipy.stats.pearsonr(y_true, y_pred)[0]
    return pr

def np_poiss(y_true, y_pred):
    return y_pred - y_true * np.log(y_pred)

def get_data(pred, cell_line=1):
    # pred = '/home/shush/profile/QuantPred/datasets/ATAC_v2/grid5/pred_i_3072_w_128_bpnet_fftmse.h5'
    title_str = pred.split('/')[-1]
    dataset = h5py.File(pred, 'r')
    test_pred = dataset['test_pred'][:]
    test_y = dataset['test_y'][:]
    df = pd.DataFrame({'true_cov': test_y.mean(axis=1)[:,cell_line],
                  'pred_cov': test_pred.mean(axis=1)[:,cell_line]})
    data = {'raw':(test_y[:,:,cell_line], test_pred[:,:,cell_line]), 'avg': df}
    return data

def create_scatter(pred=DATA_DIR, cell_line=CELL_LINE):
    data = get_data(pred, cell_line)
    fig = px.scatter(data['avg'], x='true_cov',
                     y='pred_cov', title='Average true vs pred', opacity=0.2)
        # sns.scatterplot(x=test_y.mean(axis=1)[:,t], y=test_pred.mean(axis=1)[:,t], alpha=0.1, ax=axs[t,m])
        # axs[t,m].plot([0,1],[0,1], transform=axs[t,m].transAxes, color='r')
    return fig




fig = create_scatter()


app.layout = html.Div([
    # main scatter plot panel
    html.Div([dcc.Graph(id='predictions', figure=fig)],
              style={'width': '49%', 'display': 'inline-block'}),
    html.Div([ dcc.Graph(id='profile', figure=fig)],
              style={'width': '49%', 'display': 'inline-block'})
    #
    # ])

    ])



@app.callback(
    dash.dependencies.Output('profile', 'figure'),
    [dash.dependencies.Input('predictions', 'hoverData')])
def update_profile(hoverData, pred=DATA_DIR, cell_line=CELL_LINE):
    data = get_data(pred, cell_line)
    x = hoverData['points'][0]['x']
    y = hoverData['points'][0]['y']
    seq_n = np.where(data['avg']==[x, y])[0][0]
    pr = scipy_pr(data['raw'][0][seq_n,:], data['raw'][1][seq_n,:])
    fig = px.line(x=np.arange(len(data['raw'][0][seq_n,:])),
                  y = data['raw'][0][seq_n,:], title="Coverage")
    fig.add_scatter(x=np.arange(len(data['raw'][1][seq_n,:])),
                    y = data['raw'][1][seq_n,:], name='Predicted, pr={}'.format(np.around(pr,3)))
    fig.add_scatter(x=np.arange(len(data['raw'][1][seq_n,:])),
                    y = np_mse(data['raw'][0][seq_n,:], data['raw'][1][seq_n,:]),
                    name='MSE')
    fig.add_scatter(x=np.arange(len(data['raw'][1][seq_n,:])),
                    y = np_poiss(data['raw'][0][seq_n,:], data['raw'][1][seq_n,:]),
                    name='Poisson loss')
    # fig.add_vrect( x0=0, x1=2, fillcolor="LightSalmon", opacity=0.5, layer="below", line_width=0)
    fig.update_layout()
    return fig
# df = pd.read_csv('https://plotly.github.io/datasets/country_indicators.csv')
#
# available_indicators = df['Indicator Name'].unique()
#
# app.layout = html.Div([
#     html.H1(children='Hello Dash'),
#     html.Div([
#
#         html.Div([
#             dcc.Dropdown(
#                 id='crossfilter-xaxis-column',
#                 options=[{'label': i, 'value': i} for i in available_indicators],
#                 value='Fertility rate, total (births per woman)'
#             ),
#             dcc.RadioItems(
#                 id='crossfilter-xaxis-type',
#                 options=[{'label': i, 'value': i} for i in ['Linear', 'Log']],
#                 value='Linear',
#                 labelStyle={'display': 'inline-block'}
#             )
#         ],
#         style={'width': '49%', 'display': 'inline-block'}),
#
#         html.Div([
#             dcc.Dropdown(
#                 id='crossfilter-yaxis-column',
#                 options=[{'label': i, 'value': i} for i in available_indicators],
#                 value='Life expectancy at birth, total (years)'
#             ),
#             dcc.RadioItems(
#                 id='crossfilter-yaxis-type',
#                 options=[{'label': i, 'value': i} for i in ['Linear', 'Log']],
#                 value='Linear',
#                 labelStyle={'display': 'inline-block'}
#             )
#         ], style={'width': '49%', 'float': 'right', 'display': 'inline-block'})
#     ], style={
#         'borderBottom': 'thin lightgrey solid',
#         'backgroundColor': 'rgb(250, 250, 250)',
#         'padding': '10px 5px'
#     }),
#
#     html.Div([
#         dcc.Graph(
#             id='crossfilter-indicator-scatter',
#             hoverData={'points': [{'customdata': 'Japan'}]}
#         )
#     ], style={'width': '49%', 'display': 'inline-block', 'padding': '0 20'}),
#     html.Div([
#         dcc.Graph(id='x-time-series'),
#         dcc.Graph(id='y-time-series'),
#     ], style={'display': 'inline-block', 'width': '49%'}),
#
#     html.Div(dcc.Slider(
#         id='crossfilter-year--slider',
#         min=df['Year'].min(),
#         max=df['Year'].max(),
#         value=df['Year'].max(),
#         marks={str(year): str(year) for year in df['Year'].unique()},
#         step=None
#     ), style={'width': '49%', 'padding': '0px 20px 20px 20px'})
# ])
#
#
# @app.callback(
#     dash.dependencies.Output('crossfilter-indicator-scatter', 'figure'),
#     [dash.dependencies.Input('crossfilter-xaxis-column', 'value'),
#      dash.dependencies.Input('crossfilter-yaxis-column', 'value'),
#      dash.dependencies.Input('crossfilter-xaxis-type', 'value'),
#      dash.dependencies.Input('crossfilter-yaxis-type', 'value'),
#      dash.dependencies.Input('crossfilter-year--slider', 'value')])
# def update_graph(xaxis_column_name, yaxis_column_name,
#                  xaxis_type, yaxis_type,
#                  year_value):
#     dff = df[df['Year'] == year_value]
#
#     fig = px.scatter(x=dff[dff['Indicator Name'] == xaxis_column_name]['Value'],
#             y=dff[dff['Indicator Name'] == yaxis_column_name]['Value'],
#             hover_name=dff[dff['Indicator Name'] == yaxis_column_name]['Country Name']
#             )
#
#     fig.update_traces(customdata=dff[dff['Indicator Name'] == yaxis_column_name]['Country Name'])
#
#     fig.update_xaxes(title=xaxis_column_name, type='linear' if xaxis_type == 'Linear' else 'log')
#
#     fig.update_yaxes(title=yaxis_column_name, type='linear' if yaxis_type == 'Linear' else 'log')
#
#     fig.update_layout(margin={'l': 40, 'b': 40, 't': 10, 'r': 0}, hovermode='closest')
#
#     return fig
#
#
# def create_time_series(dff, axis_type, title):
#
#     fig = px.scatter(dff, x='Year', y='Value')
#
#     fig.update_traces(mode='lines+markers')
#
#     fig.update_xaxes(showgrid=False)
#
#     fig.update_yaxes(type='linear' if axis_type == 'Linear' else 'log')
#
#     fig.add_annotation(x=0, y=0.85, xanchor='left', yanchor='bottom',
#                        xref='paper', yref='paper', showarrow=False, align='left',
#                        bgcolor='rgba(255, 255, 255, 0.5)', text=title)
#
#     fig.update_layout(height=225, margin={'l': 20, 'b': 30, 'r': 10, 't': 10})
#
#     return fig
#
#
# @app.callback(
#     dash.dependencies.Output('x-time-series', 'figure'),
#     [dash.dependencies.Input('crossfilter-indicator-scatter', 'hoverData'),
#      dash.dependencies.Input('crossfilter-xaxis-column', 'value'),
#      dash.dependencies.Input('crossfilter-xaxis-type', 'value')])
# def update_y_timeseries(hoverData, xaxis_column_name, axis_type):
#     country_name = hoverData['points'][0]['customdata']
#     dff = df[df['Country Name'] == country_name]
#     dff = dff[dff['Indicator Name'] == xaxis_column_name]
#     title = '<b>{}</b><br>{}'.format(country_name, xaxis_column_name)
#     return create_time_series(dff, axis_type, title)
#
#
# @app.callback(
#     dash.dependencies.Output('y-time-series', 'figure'),
#     [dash.dependencies.Input('crossfilter-indicator-scatter', 'hoverData'),
#      dash.dependencies.Input('crossfilter-yaxis-column', 'value'),
#      dash.dependencies.Input('crossfilter-yaxis-type', 'value')])
# def update_x_timeseries(hoverData, yaxis_column_name, axis_type):
#     dff = df[df['Country Name'] == hoverData['points'][0]['customdata']]
#     dff = dff[dff['Indicator Name'] == yaxis_column_name]
#     return create_time_series(dff, axis_type, yaxis_column_name)
#

if __name__ == '__main__':
    app.run_server(debug=True, port=8000)
