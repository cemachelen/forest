import {DatePickerView, DatePicker} from "models/widgets/date_picker"

export class CustomPickerView extends DatePickerView {
    connect_signals(): void {
        super.connect_signals()
        this.connect(this.model.properties.value.change, () => {
            let date = new Date(this.model.value)
            this.inputEl.value = date.toDateString() + " " + date.toLocaleTimeString();
        })
    }

    _on_select(date: Date): void {
        let model_date = new Date(this.model.value)
        let year = date.getFullYear()
        let month = date.getMonth()
        let day = date.getDate()
        let hour = model_date.getHours()
        let minute = model_date.getMinutes()
        let full_date = new Date(year, month, day, hour, minute)
        console.log(full_date)
        this.model.value = full_date
    }
}

export class CustomPicker extends DatePicker {
    default_view = CustomPickerView
    type = "CustomPicker"
}
