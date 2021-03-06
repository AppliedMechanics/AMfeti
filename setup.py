import sys

"""
Setup file for automatic installation of AMfe in development mode.
Run: 'python setup.py sdist' for Source Distribution
Run: 'python setup.py install' for Installation
Run: 'python setup.py bdist_wheel' for Building Binary-Wheel
    (recommended for windows-distributions)
    Attention: For every python-minor-version an extra wheel has to be built
    Use environments and install different python versions by using
    conda create -n python34 python=3.4 anaconda
Run: 'pip install wheelfile.whl' for Installing Binary-Wheel
Run: 'python setup_develop.py bdist --format=<format> für Binary-Distribution:
    <format>=gztar|ztar|tar|zip|rpm|pgktool|sdux|wininst|msi
    Recommended: tar|zip|wininst (evtl. msi)
Run: 'python setup_develop.py bdist --help-formats' to find out which distribution
    formats are available
"""

# Uncomment next line for debugging
# DISTUTILS_DEBUG='DEBUG'


def query_yes_no(question, default="yes"):
    """
    Ask a yes/no question and return their answer.

    Parameters
    ----------
    question: String
        The question to be asked

    default: String "yes" or "no"
        The default answer

    Returns
    -------
    answer: Boolean
        Answer: True if yes, False if no.
    """

    valid = {"yes": True, "y": True, "ye": True, "no": False, "n": False}
    if default is None:
        prompt = " [y/n] "
    elif default == "yes":
        prompt = " [Y/n] "
    elif default == "no":
        prompt = " [y/N] "
    else:
        raise ValueError("invalid default answer: '%s'" % default)

    while True:
        sys.stdout.write(question + prompt)
        choice = input().lower()
        if default is not None and choice == '':
            return valid[default]
        elif choice in valid:
            return valid[choice]
        else:
            sys.stdout.write("Please respond with 'yes' or 'no'.\n")


if __name__ == '__main__':
    from setuptools.config import read_configuration
    config_raw = read_configuration('./meta.cfg')
    config = dict()
    config.update(config_raw['metadata'])
    config.update(config_raw['options'])
    ext_modules = []

    from setuptools import setup
    setup(**config)
