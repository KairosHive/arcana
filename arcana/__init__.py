"""
Arcana -- latent space navigation for large image and audio collections.

This file exists to give the version one home. It used to live in two places
that had already drifted apart: pyproject.toml said 0.1.0 while the installer
script said 0.2.0, so a pip install and an installed build could report
different versions of identical code.

Everything else now reads from here:

    pyproject.toml            [tool.setuptools.dynamic] version = {attr = ...}
    installer/arcana.spec     stamps the frozen executable
    installer/arcana.iss      via installer/version.iss, written by the spec

Keep this module free of imports. setuptools reads __version__ statically when
it can, and anything heavier makes `import arcana.paths` pay for it.
"""

__version__ = "0.2.0"
