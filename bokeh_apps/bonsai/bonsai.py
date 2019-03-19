import picker
import slider
import datetime as dt


class DatetimePicker(object):
    def __init__(self):
        self.date_picker = picker.CustomPicker()
        self.hour_slider = slider.HourSlider(
            start=0,
            end=24,
            step=1,
            value=0,
            width=280,
            show_value=False)
        self.date_picker.on_change("value", self._on_picker_change)
        self.hour_slider.on_change("value", self._on_slider_change)
        self.callbacks = []
        self._old = None

    @property
    def value(self):
        if self.date_picker.value is None:
            date = dt.datetime.today().replace(minute=0, second=0)
        else:
            date = self.date_picker.value
        hour = int(self.hour_slider.value)
        return dt.datetime(date.year, date.month, date.day, hour)

    @value.setter
    def value(self, date):
        self.date_picker.value = date
        self.hour_slider.value = int(date.hour)
        self.trigger(date)

    def on_change(self, attr, callback):
        self.callbacks.append(callback)

    def trigger(self, value):
        for callback in self.callbacks:
            callback("value", self._old, value)
        self._old = value

    def _on_slider_change(self, attr, old, new):
        hour = int(new)
        self.value = self.value.replace(hour=hour)

    def _on_picker_change(self, attr, old, new):
        self.trigger(self.value)
