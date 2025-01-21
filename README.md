# kauldron

[![Unittests](https://github.com/google-research/kauldron/actions/workflows/pytest_and_autopublish.yml/badge.svg)](https://github.com/google-research/kauldron/actions/workflows/pytest_and_autopublish.yml)
[![PyPI version](https://badge.fury.io/py/kauldron.svg)](https://badge.fury.io/py/kauldron)
[![Documentation Status](https://readthedocs.org/projects/kauldron/badge/?version=latest)](https://kauldron.readthedocs.io/en/latest/?badge=latest)

Kauldron is a library for training machine learning models, optimized for
**research velocity** and **modularity**.

**Modularity**:

*   All parts of Kauldron are self-contained, so can be used independently
    outside Kauldron.
*   Use any dataset (TFDS, Grain, SeqIO, your custom pipeline),
    any (flax) model, any optimizer,... Kauldron provides the
    glue that link everything together.
*   Everything can be customized and overwritten (e.g. sweep over models
    architecture, overwrite any inner layer parameter,...)

**Research velocity**:

*   Everything should work out-of the box. The
    [example configs](https://github.com/google-research/kauldron/tree/main/examples/mnist_autoencoder.py)
    can be used and customized as a starting point.
*   Colab-first workflow for easy prototyping and fast iteration
*   Polished user experience (integrated XM plots, profiler,
    post-mortem debugging on borg, runtime shape checking, and many others...).
[Open an issue](https://github.com/google-research/kauldron/issues)..

*This is not an officially supported Google product.*
