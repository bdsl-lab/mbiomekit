import pytest
from mbiomekit import main


def test_main_help(capsys):
    with pytest.raises(SystemExit):
        main.main()