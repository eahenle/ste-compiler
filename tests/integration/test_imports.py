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
