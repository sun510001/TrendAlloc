import pytz
from datetime import datetime, time

class TimeUtilities:
    """
    Utility class for time-related calculations and timezone conversions.
    """

    @staticmethod
    def get_time_difference(hour: int = 9, minute: int = 30, region: str = "America/New_York") -> int:
        """
        Calculate the difference in minutes between the current time and a specific time in a target region.

        Args:
            hour (int): Target hour. Defaults to 9.
            minute (int): Target minute. Defaults to 30.
            region (str): Target timezone region. Defaults to "America/New_York".

        Returns:
            int: Time difference in minutes.
        """
        ny_tz = pytz.timezone(region)
        now_ny = datetime.now(ny_tz)
        target_time_ny = ny_tz.localize(datetime.combine(now_ny.date(), time(hour, minute)))
        time_difference = now_ny - target_time_ny
        return int(time_difference.total_seconds() / 60)

    @staticmethod
    def get_time_offset(datetime_obj: datetime) -> float:
        """
        Calculate the time difference in seconds between the current local time and a provided datetime object.

        Args:
            datetime_obj (datetime): The datetime object to compare against.

        Returns:
            float: Time offset in seconds.
        """
        local_time = datetime.now()
        time_difference = local_time - datetime_obj
        return time_difference.total_seconds()
