import numpy as np
import pytest  # type: ignore
import pandas as pd
from typing import List
from bk_clustering.utilities import calculation_utilities


@pytest.fixture
def get_basic_list(x=5):
    return pd.Series(list(range(x)))


def test_normalize_basic(get_basic_list):
    result = calculation_utilities.normalize(get_basic_list)
    assert np.allclose(result, [1.0, 250.75, 500.5, 750.25, 1000.0])
