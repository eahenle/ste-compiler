import subprocess
import sys


def test_validators_can_be_imported_before_public_neural_realizer(tmp_path):
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "from ste_compiler.validators import LexicalValidator; "
                "from ste_compiler.realizer import NeuralRealizer; "
                "assert LexicalValidator and NeuralRealizer"
            ),
        ],
        cwd=tmp_path,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_cli_import_and_help_do_not_require_fcntl(tmp_path):
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            """
import builtins

original_import = builtins.__import__

def import_without_fcntl(name, globals=None, locals=None, fromlist=(), level=0):
    if name == "fcntl":
        raise ModuleNotFoundError("No module named 'fcntl'", name="fcntl")
    return original_import(name, globals, locals, fromlist, level)

builtins.__import__ = import_without_fcntl
from ste_compiler.cli import app
from typer.testing import CliRunner

result = CliRunner().invoke(app, ["--help"])
assert result.exit_code == 0, result.output
""",
        ],
        cwd=tmp_path,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
