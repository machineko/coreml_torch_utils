import os
import nox



os.environ.update({"PDM_IGNORE_SAVED_PYTHON": "1"})
os.environ.pop("PYTHONPATH", None)

@nox.session
def tests(session): # pragma: no cover
    session.run("pdm", "install", "-G", "test", external=True)
    session.run("pytest", "coreml_utils/tests/", "--cov=coreml_utils", "--cov-report=xml", "-W", "ignore::DeprecationWarning")

@nox.session()
def lint(session): # pragma: no cover
    session.run("pdm", "install", "-G", "lint", external=True)
    session.run("black", "--line-length=119", "coreml_utils")