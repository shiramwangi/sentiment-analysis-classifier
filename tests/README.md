###  Running Tests

We use **pytest** for unit and integration testing.

1. Install pytest (if not present):

   ```bash
   pip install pytest
2. From the project root run
   ```bash
   pytest

   
Pytest automatically discovers test files named test_*.py in the tests/ directory. You’ll see output indicating passed/failed tests.([turn0search20], [turn0search16])
 
---

##  Summary

- Added `tests/test_preprocess.py` and `tests/test_pipeline.py` to improve reliability via automated testing.
- Enhanced `train_model.py` to save performance metrics into `results/metrics.json`.
- Included instructions for running tests using pytest.


