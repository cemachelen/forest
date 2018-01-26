from bokeh.server.server import Server
from bokeh.application import Application
from bokeh.application.handlers.directory import DirectoryHandler
from bokeh.plotting import figure, ColumnDataSource

apps = {'/': Application(DirectoryHandler(filename='documents/plot_sea_two_model_comparison'))}

server = Server(apps, port=5006)
server.start()
server.run_until_shutdown()