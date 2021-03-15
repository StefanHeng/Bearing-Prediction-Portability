from vib_record import VibRecord
from vib_transfer import VibTransfer


if __name__ == "__main__":
    rec = VibRecord()
    t = VibTransfer()
    t.degrading_diff(rec.get_feature_series(0))

