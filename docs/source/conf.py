# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "ragas"
copyright = "2023, ExplodingGradients"
author = "ExplodingGradients"
release = "0.0.16"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "myst_parser",
    "sphinxawesome_theme.highlighting",
    # "sphinxawesome_theme.docsearch",
]

templates_path = ["_templates"]
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinxawesome_theme"
html_static_path = ["_static"]

# -- sphinxawesome_theme configurations -------------------------------------------------

html_theme_options = {
    "logo_only": True,
    "logo_light": "./assets/1.jpg",
    "logo_dark": "./assets/2.jpg",
}
