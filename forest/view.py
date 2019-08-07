import bokeh.models


class Image(object):
    def __init__(self, color_mapper, tooltips=None, formatters=None):
        self.color_mapper = color_mapper
        self.source = bokeh.models.ColumnDataSource({
                "x": [],
                "y": [],
                "dw": [],
                "dh": [],
                "image": []})
        self.tooltips = tooltips
        self.formatters = formatters

    @classmethod
    def unified_model(cls, color_mapper):
        tooltips = [
                ("Name", "@name"),
                ("Value", "@image"),
                ('Length', '@length'),
                ('Valid', '@valid{%F %H:%M}'),
                ('Initial', '@initial{%F %H:%M}'),
                ("Level", "@level")]
        formatters = {
                'valid': 'datetime',
                'initial': 'datetime'}
        return cls(color_mapper, tooltips=tooltips, formatters=formatters)

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
        if self.tooltips is not None:
            hover_tool = bokeh.models.HoverTool(
                    renderers=[renderer],
                    tooltips=self.tooltips,
                    formatters=self.formatters)
            figure.add_tools(hover_tool)
        return renderer
