# Contributing code

Install the code with development and docs dependencies:

```bash
uv sync --all-groups
```

## Prior to PR:

### Format code and sort imports

```bash
black protocols
isort protocols
```

### lint code

```bash
ruff check protocols
```

### Update docs

See the [docs/README.md](docs/README.md)
