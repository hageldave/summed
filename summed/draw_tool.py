import numpy as np
from bokeh.server.server import Server
from bokeh.models import Div, ColumnDataSource, FreehandDrawTool, DataTable, TableColumn, HTMLTemplateFormatter, PolyDrawTool
from bokeh.plotting import figure
from bokeh.layouts import column, row
import projection


def create_application(doc):
    # website titles
    doc.title = "Bokeh Skeleton"
    doc.add_root(Div(text="<b>Bokeh Skeleton</b>"))
    # data

    cds = ColumnDataSource(data=dict(xs=[[]], ys=[[]]))
    # visualization
    plot = figure(
        width=500,
        height=500,
        x_range=(-100, 100),
        y_range=(-100, 100),
        output_backend="svg",
    )
    r = plot.multi_line('xs', 'ys', source = cds, line_width=3)
    tool = PolyDrawTool(renderers=[r], num_objects=1)
    plot.add_tools(tool)

    cds2 = ColumnDataSource(data=dict(xs=[[]], ys=[[]]))
    plot2 = figure(width=500, height=500, x_range=(0, 100), y_range=(-100, 100))
    r2 = plot2.multi_line('xs', 'ys', source = cds2)
    def onchangedoproj(a,b,c):
        arr = np.vstack([cds.data['xs'][0], cds.data['ys'][0]]).T
        segments = arr[1:,:]-arr[:-1,]
        segments = projection.normalize_rows(segments)
        pmat = projection.summed_dirs(segments, 2)
        arr2 = arr @ pmat

        V = np.linalg.svd(segments, full_matrices=False).Vh.T
        pmat_svd = V[:, :2]

        cds2.data = dict(ys=[arr2[:,0]], xs=[np.linspace(0,100, arr2.shape[0])])
        cds3.data = dict(xs=[[0,pmat[0,0]*50], [-pmat_svd[0,0]*50,pmat_svd[0,0]*50]], ys=[[0,pmat[1,0]*50], [-pmat_svd[1,0]*50,pmat_svd[1,0]*50]], color=['#00ff00','#ff00ff'])
    cds.on_change('data', onchangedoproj)

    cds3 = ColumnDataSource(data=dict(xs=[], ys=[], color=[]))
    plot.multi_line('xs', 'ys', source = cds3, line_color='color', line_width=3)

    # data table
    tablecols = [TableColumn(field=col, title=col, formatter=HTMLTemplateFormatter(template=f"<code><%= value.map(num => parseFloat(num.toFixed(2))) %></code>")) for col in cds.column_names]
    table = DataTable(source=cds, columns=tablecols, width=500, height=200)
    table2 = DataTable(source=cds2, columns=tablecols, width=500, height=200)
    doc.add_root(row(plot,plot2))
    doc.add_root(row(table,table2))


def start_server():
    server = Server(create_application)
    # start timers and services and immediately return
    server.start()
    server.io_loop.add_callback(server.show, "/")
    server.io_loop.start()


if __name__ == '__main__':
    start_server()