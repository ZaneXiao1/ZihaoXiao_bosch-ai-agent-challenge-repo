"""Pytest configuration — suppress noisy MLflow warnings."""
import unittest.mock as mock
import mlflow.pyfunc.utils.data_validation as _dv

# MLflow's color_warning emits ANSI-colored UserWarnings that bypass standard
# pytest warning filters. Replace it with a no-op at import time.
_dv.color_warning = mock.MagicMock()
