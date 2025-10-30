import pytest
import numpy as np
from ligotools.readligo import loaddata
import os


path = "data/L-L1_LOSC_4_V1-1126259446-32.hdf5"

def testing_shape():
    s, t, d = loaddata(path, "L1")
    assert isinstance(s, np.ndarray)
    assert isinstance(t, np.ndarray)
    assert isinstance(d, dict)


def testing_error():
    with pytest.raises(FileNotFoundError):
        loaddata("some_file.hdf5", "L1")