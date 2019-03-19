from bokeh.models import DatePicker


class CustomPicker(DatePicker):
    __implementation__ = "picker.ts"
