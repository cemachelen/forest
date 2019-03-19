import {SliderView, Slider} from "models/widgets/slider"


export class HourSliderView extends SliderView {}


export class HourSlider extends Slider {
    default_view = HourSliderView
    type = "HourSlider"

    pretty(value: number): string {
        return Math.round(value).toString().padStart(2) + ":00"
    }
}
