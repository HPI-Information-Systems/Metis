import logging

logger = logging.getLogger("metis")
logging.basicConfig(level=logging.INFO)


def warn_unconfigured_columns(
    logger: logging.Logger,
    data_columns: set[str] | list[str],
    configured_columns: set[str] | list[str],
    config_type: str,
):
    extraneous_rules = set(configured_columns) - set(data_columns)
    if extraneous_rules:
        logger.warning(
            f"The following columns have {config_type} defined but are not present in the data: {extraneous_rules}. These {config_type} will be ignored."
        )

    extraneous_columns = set(data_columns) - set(configured_columns)
    if extraneous_columns:
        logger.warning(
            f"The following columns are present in the data but have no {config_type} defined: {extraneous_columns}. These columns will be skipped."
        )
