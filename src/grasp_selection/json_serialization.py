"""
Module to serialize and deserialize JSON with numpy arrays. Adapted from
http://stackoverflow.com/a/24375113/723090 so that arrays are human-readable.

Author: Brian Hou
"""

import json as _json
import numpy as np

class NumpyEncoder(_json.JSONEncoder):
    def default(self, obj):
        """
        If obj is an ndarray it will be converted into a dict with dtype, shape
        and the data as a list.
        """
        if isinstance(obj, np.ndarray):
            return dict(__ndarray__=obj.tolist(),
                        dtype=str(obj.dtype),
                        shape=obj.shape)
        # Let the base class default method raise the TypeError
        return _json.JSONEncoder(self, obj)

def json_numpy_obj_hook(dct):
    """
    Decodes a previously encoded numpy ndarray with proper shape and dtype.
    """
    if isinstance(dct, dict) and '__ndarray__' in dct:
        data = np.asarray(dct['__ndarray__'], dtype=dct['dtype'])
        return data.reshape(dct['shape'])
    return dct

def dump(*args, **kwargs):
    kwargs.update(dict(cls=NumpyEncoder,
                       sort_keys=True,
                       indent=4,
                       separators=(',', ': ')))
    return _json.dump(*args, **kwargs)

def load(*args, **kwargs):
    kwargs.update(dict(object_hook=json_numpy_obj_hook))
    return _json.load(*args, **kwargs)
