from bokeh.models import Slider


class HourSlider(Slider):
    __implementation__ = "hour_slider.ts"
