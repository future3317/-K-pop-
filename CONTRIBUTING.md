# Contributing

Thanks for contributing to KPopScope.

## Development setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
pytest -q
```

## Data policy

Please do not commit copyrighted music files or dataset dumps. Use small synthetic audio for tests, or provide scripts that users can run after accepting each dataset license.

## Suggested issues

- Add downbeat tracking.
- Improve segment labeling.
- Add trained K-pop tag classifier checkpoints.
- Add HTML report export.
- Add user-facing annotation tool for K-pop tags.
