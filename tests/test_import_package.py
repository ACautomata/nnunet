import nnunet


def test_import_package() -> None:
    assert nnunet.__doc__ is not None
