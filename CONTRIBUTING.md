# Contributing

Thanks for contributing to this project.

## Getting Started

1. Fork the repository and create a branch from main.
2. Use Python 3.11.
3. Install dependencies:

   pip install -r requirements.txt

4. Run tests before opening a PR:

   python -m pytest tests -vv -ra

## Pull Request Guidelines

1. Keep changes scoped to one concern.
2. Update README or docs when behavior changes.
3. Add or update tests for bug fixes and new logic where practical.
4. Ensure CI is passing.

## CARLA-Specific Notes

1. Use DX12 launch settings for RTX 5080 workflows.
2. Do not introduce runtime map switching in code paths that call CARLA.
3. Keep generated data, model checkpoints, and logs out of git.
