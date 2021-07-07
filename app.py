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
import plotly.graph_objects as go
from metrics import *

# y axis fixed somewhere in the Average
# remove mse
# dropdown menu for cell lines
# plot extra cell lines

DATA_DIR='/home/shush/profile/QuantPred/compare_training/1v2_custom_test/pred.h5'
CELL_LINE=1
server = flask.Flask(__name__)
app = dash.Dash(__name__, server=server)


def get_data(pred, cell_line=1):
    # pred = '/home/shush/profile/QuantPred/datasets/ATAC_v2/grid5/pred_i_3072_w_128_bpnet_fftmse.h5'
    title_str = pred.split('/')[-1]
    dataset = h5py.File(pred, 'r')
    test_pred = dataset['test_pred'][:]
    test_y = dataset['test_y'][:]
    coords = dataset['coords'][:]

    # good_idx = np.where(test_y[:,:,cell_line].max(axis=1)>=2)
    colors = np.array([test_y[:,:,cell_line].max(axis=1)>=2][0], dtype=int)
    # test_pred = test_pred[good_idx][:,:,cell_line]
    # test_y = test_y[good_idx][:,:,cell_line]
    # coords = coords[good_idx]

    pr_all = [scipy_pr(test_y[seq_n,:,cell_line], test_pred[seq_n,:,cell_line]) for seq_n in range(test_y.shape[0])]
    df = pd.DataFrame({'true_cov': test_y.mean(axis=1)[:,cell_line],
                  'pred_cov': test_pred.mean(axis=1)[:,cell_line]})
    data = {'raw':(test_y[:,:,cell_line], test_pred[:,:,cell_line]),
            'coords': coords, 'avg': df, 'pr_all': pr_all, 'peak': colors}
    return data

def create_scatter(pred=DATA_DIR, cell_line=CELL_LINE):
    data = get_data(pred, cell_line)
    # x_min = np.min(data['avg']['true_cov'])
    # x_max = np.max(data['avg']['true_cov'])
    # y_min = np.min(data['avg']['pred_cov'])
    # y_max = np.max(data['avg']['pred_cov'])
    # data_and_pr = data['avg']['pearsonr'] = data['pr_all']
    data['avg']['peak'] = data['peak'].astype(str)
    fig = px.scatter(data['avg'], x='true_cov',
                     y='pred_cov', title='Average true vs pred', opacity=0.6,
                     color='peak', color_continuous_scale='viridis')
    fig.add_scatter(x=np.arange(0, 4), y=np.arange(0, 4), mode='lines', hoverinfo='skip', name='')
        # sns.scatterplot(x=test_y.mean(axis=1)[:,t], y=test_pred.mean(axis=1)[:,t], alpha=0.1, ax=axs[t,m])
        # axs[t,m].plot([0,1],[0,1], transform=axs[t,m].transAxes, color='r')
    # fig.update_layout(coloraxis_colorbar=dict(yanchor="top", y=0, x=-1,
    #                                       ticks="outside"))

    return fig




fig = create_scatter()


app.layout = html.Div([
    # main scatter plot panel
    html.Div([dcc.Graph(id='predictions', figure=fig)],
              style={'width': '49%', 'display': 'inline-block'}),
    html.Div([
              dcc.Graph(id='profile')],
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
    sc = scipy_sc(data['raw'][0][seq_n,:], data['raw'][1][seq_n,:])
    fig = px.line(x=np.arange(len(data['raw'][0][seq_n,:])),
                  y = data['raw'][0][seq_n,:], title="Coverage, Pearson R={}, Spearman corr={}, {}".format(np.around(pr,3),
                                                                                                           np.around(sc,3),
                                                                                                           'chr'+'_'.join(data['coords'][:][seq_n,:].astype(np.str))))
    fig.add_scatter(x=np.arange(len(data['raw'][1][seq_n,:])),
                    y = data['raw'][1][seq_n,:],
                    name='Predicted', mode='lines', line=go.scatter.Line(color="red"))
    # fig.add_vrect( x0=0, x1=2, fillcolor="LightSalmon", opacity=0.5, layer="below", line_width=0)
    fig.update_layout()
    return fig

# @app.callback(
#     dash.dependencies.Output('mse', 'figure'),
#     [dash.dependencies.Input('predictions', 'hoverData')])
# def update_profile(hoverData, pred=DATA_DIR, cell_line=CELL_LINE):
#     data = get_data(pred, cell_line)
#     x = hoverData['points'][0]['x']
#     y = hoverData['points'][0]['y']
#     seq_n = np.where(data['avg']==[x, y])[0][0]
#     pr = scipy_pr(data['raw'][0][seq_n,:], data['raw'][1][seq_n,:])
#     sc = scipy_sc(data['raw'][0][seq_n,:], data['raw'][1][seq_n,:])
#     fig = px.line(x=np.arange(len(data['raw'][1][seq_n,:])),
#                     y = np_mse(data['raw'][0][seq_n,:], data['raw'][1][seq_n,:]),
#                     title='MSE')
#     fig.update_traces(line_color='#60095C')
    # fig.add_scatter(x=np.arange(len(data['raw'][1][seq_n,:])),
    #                 y = np_poiss(data['raw'][0][seq_n,:], data['raw'][1][seq_n,:]),
    #                 name='Poisson loss')
    # fig.add_vrect( x0=0, x1=2, fillcolor="LightSalmon", opacity=0.5, layer="below", line_width=0)
    # fig.update_layout()
    # return fig

# @app.callback(
#     dash.dependencies.Output('poisson', 'figure'),
#     [dash.dependencies.Input('predictions', 'hoverData')])
# def update_profile(hoverData, pred=DATA_DIR, cell_line=CELL_LINE):
#     data = get_data(pred, cell_line)
#     x = hoverData['points'][0]['x']
#     y = hoverData['points'][0]['y']
#     seq_n = np.where(data['avg']==[x, y])[0][0]
#     pr = scipy_pr(data['raw'][0][seq_n,:], data['raw'][1][seq_n,:])
#     sc = scipy_sc(data['raw'][0][seq_n,:], data['raw'][1][seq_n,:])
#     fig = px.line(x=np.arange(len(data['raw'][1][seq_n,:])),
#                     y = np_poiss(data['raw'][0][seq_n,:], data['raw'][1][seq_n,:]),
#                     title='Poisson')
    # fig.add_scatter(x=np.arange(len(data['raw'][1][seq_n,:])),
    #                 y = np_poiss(data['raw'][0][seq_n,:], data['raw'][1][seq_n,:]),
    #                 name='Poisson loss')
    # fig.add_vrect( x0=0, x1=2, fillcolor="LightSalmon", opacity=0.5, layer="below", line_width=0)
    fig.update_layout()
    return fig



if __name__ == '__main__':
    app.run_server(debug=True, port=8000)
