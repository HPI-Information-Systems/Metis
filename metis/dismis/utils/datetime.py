from typing import Tuple, Optional
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import re

def parse_to_python_datetime(series: pd.Series) -> pd.Series:
    """
    Robustly parses a series to Python datetime objects (not pandas datetime64).
    Python datetime supports years 1-9999 without nanosecond overflow issues.
    
    Handles:
    - Numeric YYYYMMDD (8 digits)
    - Numeric YYYY (4 digits)
    - String dates in various formats
    """
    def parse_value(val):
        if pd.isna(val):
            return None
        
        # If already datetime
        if isinstance(val, (datetime, pd.Timestamp)):
            return val.to_pydatetime() if isinstance(val, pd.Timestamp) else val
        
        # If numeric, check for date formats
        if isinstance(val, (int, float, np.integer, np.floating)):
            val_int = int(val)
            # YYYYMMDD format (8 digits)
            if 10000000 <= val_int <= 99999999:
                year = val_int // 10000
                month = (val_int // 100) % 100
                day = val_int % 100
                try:
                    return datetime(year, month, day)
                except ValueError:
                    return None
            # YYYY format (4 digits)
            elif 1 <= val_int <= 9999:
                try:
                    return datetime(val_int, 1, 1)
                except ValueError:
                    return None
        
        # Try parsing as string
        str_val = str(val).strip()
        
        # Try YYYY format first (for historical dates)
        if len(str_val) == 4 and str_val.isdigit():
            try:
                year = int(str_val)
                if 1 <= year <= 9999:
                    return datetime(year, 1, 1)
            except ValueError:
                pass
        
        # Try pandas datetime parsing (handles many formats)
        try:
            pd_dt = pd.to_datetime(str_val, errors='coerce')
            if pd.notna(pd_dt):
                return pd_dt.to_pydatetime()
        except:
            pass
        
        # Apply fallback only if no alphabetical characters present
        if not re.search(r'[a-zA-Z]', str_val):
            # Fallback 1: Extract first 4-digit substring and interpret as year
            match = re.search(r'\d{4}', str_val)
            if match:
                try:
                    year = int(match.group())
                    if 1 <= year <= 9999:
                        return datetime(year, 1, 1)
                except ValueError:
                    pass
            
            # Fallback 2: Extract first 2-digit substring and interpret as 19xx
            match = re.search(r'\d{2}', str_val)
            if match:
                try:
                    year_suffix = int(match.group())
                    year = 1900 + year_suffix
                    if 1 <= year <= 9999:
                        return datetime(year, 1, 1)
                except ValueError:
                    pass
        
        return None
    
    return series.apply(parse_value)

def detect_datetime_precision(series: pd.Series) -> str:
    """
    Detects the precision of a datetime series.
    Returns one of: 'Y' (years), 'M' (months), 'd' (days), 'h' (hours), 'min' (minutes), 's' (seconds).
    """
    # Drop missing values
    valid = series.dropna()
    if len(valid) == 0:
        return "unknown"
    
    # Check precision based on datetime components
    has_seconds = any(dt.second != 0 for dt in valid if dt is not None)
    if has_seconds:
        return "s"
    
    has_minutes = any(dt.minute != 0 for dt in valid if dt is not None)
    if has_minutes:
        return "min"
    
    has_hours = any(dt.hour != 0 for dt in valid if dt is not None)
    if has_hours:
        return "h"
    
    has_days = any(dt.day != 1 for dt in valid if dt is not None)
    if has_days:
        return "d"
    
    has_months = any(dt.month != 1 for dt in valid if dt is not None)
    if has_months:
        return "M"
    
    return "Y"

def datetime_to_numeric(series: pd.Series) -> Tuple[pd.Series, Optional[datetime], str]:
    """
    Converts a series to numeric values normalized by the detected granularity.
    Uses Python datetime objects internally to avoid pandas datetime64 overflow issues.
    
    Returns:
        - numeric_series: Normalized numeric values (relative to minimum)
        - min_datetime: Minimum reference datetime
        - unit: Detected granularity ('Y', 'M', 'd', 'h', 'min', 's')
    """
    # Parse to Python datetime objects
    dt_series = parse_to_python_datetime(series)
    
    # Detect precision
    unit = detect_datetime_precision(dt_series)
    if unit == "unknown":
        return series, None, unit
    
    # Get minimum datetime
    valid = dt_series.dropna()
    if len(valid) == 0:
        return dt_series, None, unit
    
    min_dt = min(dt for dt in valid if dt is not None)
    
    # Convert to numeric based on unit
    if unit == "Y":
        numeric = dt_series.apply(lambda x: x.year - min_dt.year if not pd.isnull(x) else np.nan)
    elif unit == "M":
        numeric = dt_series.apply(
            lambda x: (x.year - min_dt.year) * 12 + (x.month - min_dt.month) if not pd.isnull(x) else np.nan
        )
    elif unit == "d":
        # Use toordinal() - days since year 1
        min_ordinal = min_dt.toordinal()
        numeric = dt_series.apply(lambda x: x.toordinal() - min_ordinal if not pd.isnull(x) else np.nan)
    elif unit == "h":
        # Total hours
        numeric = dt_series.apply(
            lambda x: (x - min_dt).total_seconds() / 3600 if not pd.isnull(x) else np.nan
        )
    elif unit == "min":
        # Total minutes
        numeric = dt_series.apply(
            lambda x: (x - min_dt).total_seconds() / 60 if not pd.isnull(x) else np.nan
        )
    elif unit == "s":
        # Total seconds
        numeric = dt_series.apply(
            lambda x: (x - min_dt).total_seconds() if not pd.isnull(x) else np.nan
        )
    else:
        return series, None, "unknown"
    
    return numeric, min_dt, unit

def numeric_to_datetime(numeric_series: pd.Series, min_datetime: datetime, unit: str) -> pd.Series:
    """
    Converts normalized numeric values back to Python datetime objects.
    
    Args:
        numeric_series: Normalized numeric values
        min_datetime: Reference datetime
        unit: Granularity ('Y', 'M', 'd', 'h', 'min', 's')
    
    Returns:
        Series of Python datetime objects
    """
    if unit not in ["d", "h", "min", "s", "M", "Y"]:
        raise ValueError("Unit must be one of 'd', 'h', 'min', 's', 'M', 'Y'")
    
    def add_time(value):
        if pd.isna(value):
            return None
        
        val = int(value) if unit in ["Y", "M", "d"] else float(value)
        
        if unit == "Y":
            return datetime(min_datetime.year + val, min_datetime.month, min_datetime.day,
                          min_datetime.hour, min_datetime.minute, min_datetime.second)
        elif unit == "M":
            # Calculate year and month
            total_months = min_datetime.year * 12 + min_datetime.month - 1 + val
            year = total_months // 12
            month = (total_months % 12) + 1
            return datetime(year, month, min_datetime.day,
                          min_datetime.hour, min_datetime.minute, min_datetime.second)
        elif unit == "d":
            min_ordinal = min_datetime.toordinal()
            return datetime.fromordinal(min_ordinal + val)
        elif unit == "h":
            return min_datetime + timedelta(hours=val)
        elif unit == "min":
            return min_datetime + timedelta(minutes=val)
        elif unit == "s":
            return min_datetime + timedelta(seconds=val)
    
    return numeric_series.apply(add_time)