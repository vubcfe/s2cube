# Contributing

Bug reports, questions and pull requests are welcome on the
[issue tracker](https://github.com/vubcfe/s2cube/issues).

## Development setup

```bash
git clone https://github.com/vubcfe/s2cube && cd s2cube
pip install -e ".[dev]"
pytest                 # offline tests
pytest -m network      # tests against the live STAC catalogues
ruff check src tests   # style
mkdocs serve           # documentation preview
```

## Guidelines

- Every new feature comes with a test; offline tests use the synthetic scenes of
  `tests/conftest.py` and must not need internet access.
- Keep the HDF5 layout backwards compatible; document any addition in `docs/data_format.md`
  and `CHANGELOG.md`.
- Start every source file with the SPDX licence header used in the existing files.
- Be respectful; we follow the [Contributor Covenant](https://www.contributor-covenant.org/version/2/1/code_of_conduct/).
