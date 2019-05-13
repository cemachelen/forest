import pandas as pd


def read(csv_files):
    if isinstance(csv_files, str):
        csv_files = [csv_files]
    frames = []
    for csv_file in csv_files:
        frame = pd.read_csv(
            csv_file,
            parse_dates=[1],
            converters={0: flash_type},
            usecols=[0, 1, 2, 3],
            names=["flash_type", "date", "longitude", "latitude"],
            header=None)
        frames.append(frame)
    frame = pd.concat(frames, ignore_index=True)
    return frame.set_index('date')


def flash_type(value):
    return {
        "0": "CG",
        "1": "IC",
        "9": "Keep alive"
    }.get(value, value)
