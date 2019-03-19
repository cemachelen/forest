import {DatePickerView, DatePicker} from "models/widgets/date_picker"

export class CustomPickerView extends DatePickerView {
    connect_signals(): void {
        super.connect_signals()
        this.connect(this.model.properties.value.change, () => {
            let date = new Date(this.model.value)
            this.inputEl.value = date.toDateString();
        })
    }
}

export class CustomPicker extends DatePicker {
    default_view = CustomPickerView
    type = "CustomPicker"
}
