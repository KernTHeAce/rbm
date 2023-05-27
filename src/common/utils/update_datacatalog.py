from typing import Any, Dict

from kedro.io import DataCatalog, MemoryDataSet


def update_datacatalog(datacatalog: DataCatalog, new_data: Dict[str, Any], replace=False, force=True):
    for key, value in new_data.items():
        if force:
            datacatalog.add(key, MemoryDataSet(value, copy_mode="assign"), replace=replace)
        else:
            datacatalog.datasets.__getattribute__(key)
            # try:
            #     datacatalog.datasets.__getattribute__(key)
            # except AttributeError:
            #     datacatalog.add(key, MemoryDataSet(value), replace=replace)

    return datacatalog
