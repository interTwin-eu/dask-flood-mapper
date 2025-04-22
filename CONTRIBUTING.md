# Contributing

Thank you for the interest in contributing to this project! We accept
contributions in the following areas:

- Reporting a bug
- Discussing the current state of the code
- Submitting a fix
- Proposing new features

## Development with GitHub

We use GitHub to host code, to track issues and feature requests, as well as
accept pull requests.

## Propose changes with Pull Requests

Pull Requests(PR) are the best way to propose changes to the codebase. Before
issuing the PR, make sure that:

1. Fork the repository and create your branch from main.
2. If you have added code that should be tested, add tests.
3. If you have changed APIs, update the documentation.
4. Ensure the test suite passes.
5. Make sure your code lints.
6. Issue that pull request!

## Contribution License

Any contribution made will be under the same license that covers the project,
the [MIT License](LICENSE.txt).

## Report bugs using Github's issues

We use GitHub issues to track public bugs. If you want to report a bug, please
open a new issue.

### How to write issues

Please report bugs with detail, background and sample code. Ideally the issue
should have:

- A quick summary and/or background
- Steps to reproduce
  - Be specific!
  - Give sample code, if possible
- What you expected would happen
- What actually happens
- Notes (possibly including why you think this might be happening, or things
  that you tried and didn't work)

## Reference

This section was adapted from
[this guideline](https://gist.github.com/briandk/3d2e8b3ec8daf5a27a62).

## Development

### Environment management

For convenience one can use `pipenv` and the `Pipfile.lock` to deterministically
install all packages used during development.

```bash
pipenv sync -d
```

Checkout the [documentation](https://pipenv.pypa.io/en/latest/) for more help
with installing pipenv and reconstructing the development environment.

### Testing

Running the requires `pytest` and is executed as follows:

```bash
# pipenv install pytest
pytest ./tests/
```

### Linting and formatting

The pre-commit hooks can be used to check whether you contribution follows the
standards as adhered to in this project. Install and activate the `pre-commit`
hooks, like so:

```bash
# pipenv install pre-commit
pre-commit install
```

Before each commiting, Ruff is actioned. To run Ruff without commiting, run:

```bash
pre-commit run --all-files
```

or

```bash
# pipenv install ruff
ruff check --fix --output-format concise
```

To fix the format, Ruff also offers this option with the command:

```bash
ruff format
```

To check whether the output of the notebook cells is removed one can do the
following:

```bash
# pipenv install nbstripout
find . -name '*.ipynb' -exec nbstripout {} +
```
