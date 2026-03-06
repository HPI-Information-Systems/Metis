from typing import Union, Optional, Dict, List
import pandas as pd


DOMAIN_PATTERNS: Dict[str, str] = {
    "email": r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$',
    "url": r'^https?://[^\s/$.?#]+\.[^\s]*$',
    "ssn": r'^(?!000|666|9\d{2})\d{3}-(?!00)\d{2}-(?!0000)\d{4}$',
    "date_iso": r'^\d{4}-(0[1-9]|1[0-2])-(0[1-9]|[12]\d|3[01])$',
    "time": r'^([01]\d|2[0-3]):[0-5]\d(:[0-5]\d)?$',
    "ip_address": r'^(?:(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\.){3}(?:25[0-5]|2[0-4]\d|[01]?\d\d?)$', 
    "zip_code": r'^\d{5}(-\d{4})?$',
    "credit_card": r'^(?:\d[ -]*){13,19}$',
    "phone": r'^(\+\d{1,3}[-.\s]?\(?\d{1,4}\)?[-.\s]?\d{1,4}[-.\s]?\d{1,9}|\(\d{3}\)[-.\s]?\d{3}[-.\s]?\d{4}|\d{3}[-.\s]\d{3}[-.\s]\d{4})$',
    "currency": r'^[($€£¥₹-]?[$€£¥₹]?\s?-?\d{1,3}(?:[,.\s]\d{3})*(?:[,.]\d{1,2})?\s?[$€£¥₹]?[)]?$',
}

NAME_INDICATORS: Dict[str, List[str]] = {
    "first_name": ["firstname", "fname", "givenname"],
    "last_name": ["lastname", "lname", "surname", "familyname"],
    "full_name": ["fullname"],
    "city": ["city", "town", "municipality"],
    "state": ["state", "province", "region"],
    "country": ["country", "nation"],
    "address": ["address", "street", "addr"],
    "postal_code": ["postal", "postcode", "zipcode"],
}


def _detect_domain_by_pattern(series: pd.Series, threshold: float = 0.8) -> Optional[str]:
    """
    Detect domain by matching against known patterns.

    :param series: Input Series.
    :param threshold: Minimum proportion of values that must match a pattern.
    :return: Domain name if detected, None otherwise.
    """
    clean_data = series.dropna()
    
    if len(clean_data) == 0:
        return None

    sample_size = min(100, len(clean_data))
    sample = clean_data.sample(n=sample_size, random_state=42).astype(str)

    for domain, pattern in DOMAIN_PATTERNS.items():
        match_ratio = sample.str.match(pattern, na=False).mean()
        if match_ratio >= threshold:
            return domain

    return None


def _detect_domain_by_column_name(column_name: str) -> Optional[str]:
    """
    Detect domain by matching column name against known indicators.

    :param column_name: Name of the column.
    :return: Domain name if detected, None otherwise.
    """
    column_normalized = column_name.lower().replace('_', '').replace(' ', '').replace('-', '')

    for domain, indicators in NAME_INDICATORS.items():
        for indicator in indicators:
            if column_normalized == indicator or column_normalized.endswith(indicator):
                return domain

    return None


def _classify_domain(series: pd.Series, column_name: Optional[str] = None) -> str:
    """
    Classify the semantic domain of a Series.

    :param series: Input Series.
    :param column_name: Optional column name for additional context.
    :return: Domain classification as string.
    """
    clean_data = series.dropna()

    if len(clean_data) == 0:
        return "unknown"

    domain_by_pattern = _detect_domain_by_pattern(clean_data)
    if domain_by_pattern:
        return domain_by_pattern

    if column_name:
        domain_by_name = _detect_domain_by_column_name(column_name)
        if domain_by_name:
            return domain_by_name

    return "unknown"


def domain(data: Union[pd.Series, pd.DataFrame]) -> Union[str, pd.Series]:
    """
    Classify the semantic domain of columns.

    Attempts to identify specific domains such as: email, phone, url, ip_address,
    credit_card, ssn, zip_code, date_iso, time, currency, first_name, last_name,
    full_name, city, state, country, address, postal_code.

    Uses a combination of pattern matching on data values and column name analysis.
    Pattern matching takes priority over column name inference.
    Returns "unknown" if no domain can be confidently identified.

    :param data: Input Series (single column) or DataFrame (multiple columns).
    :return: Domain name as string if Series input, Series of domain names if
             DataFrame input.
    """
    if isinstance(data, pd.Series):
        column_name = data.name if hasattr(data, 'name') else None
        return _classify_domain(data, column_name)
    else:
        result = {}
        for col in data.columns:
            result[col] = _classify_domain(data[col], col)
        return pd.Series(result)