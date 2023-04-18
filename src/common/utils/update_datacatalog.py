from typing import Any, Dict

from kedro.io import DataCatalog, MemoryDataSet


def update_datacatalog(datacatalog: DataCatalog, new_data: Dict[str, Any], replace=False):
    for key, value in new_data.items():
        datacatalog.add(key, MemoryDataSet(value), replace=replace)
    return datacatalog
