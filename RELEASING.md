# Releasing arcope

This project publishes to PyPI via GitHub Actions using OIDC trusted publishing —
no long-lived API tokens required.

## One-time setup (maintainer)

1. Create the project on PyPI (reserves the name):
   ```bash
   pip install build twine
   python -m build
   twine upload --repository testpypi dist/*
   ```
   Use a TestPyPI API token for the very first upload, then delete the token.

2. Configure trusted publishing on both PyPI and TestPyPI:
   - Project: `arcope`
   - Owner: `MarcYin`
   - Repository: `ARCOPE`
   - Workflow: `release.yml`
   - Environment: `pypi` (or `testpypi`)

3. Configure the `pypi` and `testpypi` environments in
   [GitHub repo settings](https://github.com/MarcYin/ARCOPE/settings/environments)
   with required reviewers or deployment branch rules if desired.

## Cutting a release

1. Update `CHANGELOG.md` under the `[Unreleased]` section and move notes
   under a new version heading.

2. Commit and tag the release:
   ```bash
   git commit -am "Release v0.1.0"
   git tag -a v0.1.0 -m "v0.1.0"
   git push origin main --tags
   ```

3. The `Release` workflow builds the sdist and wheel, runs `twine check`,
   then publishes to PyPI automatically because the tag matches `v*`.

4. Verify:
   - PyPI project page shows the new version
   - `pip install arcope==0.1.0` works in a fresh environment

## Dry-run to TestPyPI

To test the release pipeline without affecting production PyPI:

1. Go to [Actions -> Release](https://github.com/MarcYin/ARCOPE/actions/workflows/release.yml).
2. Click **Run workflow**, pick `testpypi`, confirm.
3. Check the TestPyPI project page and try installing from there:
   ```bash
   pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ arcope
   ```

## Version scheme

Versions come from git tags via `setuptools-scm`. Between tagged releases,
the version is computed as `X.Y.Z.postN+...` (with `local_scheme="no-local-version"`
so the uploaded distributions keep a PEP 440-compliant post-release number).

To build locally and inspect the version:

```bash
pip install -e .
python -c "import arc_scope; print(arc_scope.__version__)"
```
