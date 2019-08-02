import bokeh.models


class Image(object):
    def __init__(self, color_mapper, hover_tool=None):
        self.color_mapper = color_mapper
        self.source = bokeh.models.ColumnDataSource({
                "x": [],
                "y": [],
                "dw": [],
                "dh": [],
                "image": []})
        self.hover_tool = hover_tool

    @classmethod
    def unified_model(cls):
        hover_tool = bokeh.models.HoverTool(
                renderers=[renderer],
                tooltips=[
                    ("Name", "@name"),
                    ("Value", "@image"),
                    ('Length', '@length'),
                    ('Valid', '@valid{%F %H:%M}'),
                    ('Initial', '@initial{%F %H:%M}'),
                    ("Level", "@level")],
                formatters={
                    'valid': 'datetime',
                    'initial': 'datetime'
                })
        return cls(color_mapper, hover_tool)

    def render(self, data):
        self.source.data = data

    def add_figure(self, figure):
        renderer = figure.image(
                x="x",
                y="y",
                dw="dw",
                dh="dh",
                image="image",
                source=self.source,
                color_mapper=self.color_mapper)
        if self.hover_tool is not None:
            figure.add_tools(self.hover_tool)
        return renderer
