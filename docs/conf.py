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
    "sphinx_design",
    # "myst_parser",
    "sphinxawesome_theme.highlighting",
    # "sphinxawesome_theme.docsearch",
]
source_suffix = [".rst", ".md"]

templates_path = ["_templates"]
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_title = "Ragas"
html_theme = "sphinxawesome_theme"
html_static_path = ["_static"]
html_css_files = ["css/ragas.css"]
html_favicon = "./_static/favicon.ico"

html_theme_options = {
    "logo_light": "./_static/imgs/ragas-logo.png",
    "logo_dark": "./_static/imgs/ragas-logo.png",
}

# -- Myst NB Config -------------------------------------------------
nb_execution_mode = "auto"
