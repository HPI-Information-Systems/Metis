import json
from typing import Dict

class DataConfig:
    def __init__(
        self,
        config: Dict,
    ):
        self.name = config.get('name')
        self.file_name = config.get('file_name')
        self.loader = config.get('loader')
        self.delimiter = config.get('delimiter', ',')
        self.encoding = config.get('encoding', 'utf-8')
        self.header = config.get('header', 0)
        self.nrows = config.get('nrows', None)
        self.usecols = config.get('usecols', None)
        self.parse_dates = config.get('parse_dates', False)
        self.decimals = config.get('decimals', ".")
        self.thousands = config.get('thousands', None)