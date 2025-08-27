import matplotlib.pyplot as plt  # type: ignore

from ..data.smoking_record_class import SmokingRecord, NonSmokerRecord, SmokerRecord


def add_smoking_status_background(
    axis: plt.Axes, smoking_record: SmokingRecord, text_height: float, max_x: float
) -> None:
    """Add coloured background to show smoking period."""
    if isinstance(smoking_record, NonSmokerRecord):
        return
    assert smoking_record.start_smoking_age is not None
    left_edge = smoking_record.start_smoking_age
    if isinstance(smoking_record, SmokerRecord):
        right_edge = max_x + 1
    else:
        assert smoking_record.stop_smoking_age is not None
        right_edge = min(smoking_record.stop_smoking_age, max_x + 1)
    axis.axvspan(left_edge, right_edge, facecolor="green", alpha=0.2)
    # add text centred at the middle of the smoking period
    axis.text(
        (left_edge + right_edge) / 2,
        text_height,
        "smoking period",
        horizontalalignment="center",
        verticalalignment="center",
        color="green",
        alpha=0.8,
    )
