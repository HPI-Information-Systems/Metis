import datetime
from typing import Literal

from dateutil import parser

DTPrecision = Literal["year", "month", "day", "hour", "minute", "second", "microsecond"]


class datetimespy(datetime.datetime):
    def replace(self, *args, **kwargs):
        self._replaced_args = args
        self._replaced_kwargs = kwargs
        return super().replace(*args, **kwargs)


def determine_datetime_precision(dt_str: str) -> DTPrecision:
    default = datetimespy.now()
    parser.parse(dt_str, default=default)

    replaced_fields = getattr(default, "_replaced_kwargs", {})

    if "microsecond" in replaced_fields:
        return "microsecond"
    if "second" in replaced_fields:
        return "second"
    if "minute" in replaced_fields:
        return "minute"
    if "hour" in replaced_fields:
        return "hour"
    if "day" in replaced_fields:
        return "day"
    if "month" in replaced_fields:
        return "month"
    return "year"
