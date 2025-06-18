import pytest
from my_package import main


def test_main_help(capsys):
    with pytest.raises(SystemExit):
        main(["--help"])