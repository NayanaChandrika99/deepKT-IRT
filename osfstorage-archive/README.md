#### Quick start

[Install miniconda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/macos.html)

```bash
# install the virtual environment
conda env create -f env.yml


# install dependencies using poetry
poetry install

# running tests: this only works after setting the tests/conftest/raw_data_dir and tests/conftest/processed_data_dir
pytest tests
```
