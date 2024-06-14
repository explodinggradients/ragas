from dataclasses import asdict

from sphinxawesome_theme import ThemeOptions

from ragas import __version__

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "ragas"
copyright = "2023, ExplodingGradients"
author = "ExplodingGradients"
release = "0.0.16"
print("RAGAS VERSION", __version__)

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

html_title = "Ragas"
html_theme = "sphinxawesome_theme"
html_static_path = ["_static"]
html_css_files = [
    "css/ragas.css",
    "css/highlight_python_dark.css",
    "css/highlight_ipython3_dark.css",
]
html_js_files = ["js/mendable_chat_bubble.js", "js/octolane.js"]
html_favicon = "./_static/favicon.ico"

extensions = [
    "sphinx.ext.doctest",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.autosummary",
    "sphinx_design",
    "sphinxawesome_theme.highlighting",
    # "sphinxawesome_theme.docsearch",
    "myst_nb",
]

source_suffix = [".rst", ".md"]
templates_path = ["_templates"]
exclude_patterns = ["_build"]
myst_number_code_blocks = ["typescript"]

# algolia search
# docsearch_app_id = os.getenv("DOCSEARCH_APP_ID")
# docsearch_api_key = os.getenv("DOCSEARCH_API_KEY")
# docsearch_index_name = os.getenv("DOCSEARCH_INDEX_NAME")

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

# enable extensions
# https://myst-parser.readthedocs.io/en/latest/syntax/optional.html#syntax-admonitions
myst_enable_extensions = [
    "amsmath",
    "colon_fence",
    "deflist",
    "dollarmath",
    "fieldlist",
    "html_admonition",
    "html_image",
    "linkify",
    "replacements",
    "smartquotes",
    "strikethrough",
    "substitution",
    "tasklist",
    # "attrs_block",
]

theme_options = ThemeOptions(
    logo_light="./_static/imgs/ragas-logo.png",
    logo_dark="./_static/imgs/ragas-logo.png",
    extra_header_link_icons={
        "discord": {
            "link": "https://discord.gg/5djav8GGNZ",
            "icon": """<svg height="26px" style="margin-right: 3px; display: inline" viewBox="0 0 22 22" fill="currentColor" xmlns="http://www.w3.org/2000/svg"><path fill-rule="evenod" clip-rule="evenod" d="M18.59 5.88997C17.36 5.31997 16.05 4.89997 14.67 4.65997C14.5 4.95997 14.3 5.36997 14.17 5.69997C12.71 5.47997 11.26 5.47997 9.83001 5.69997C9.69001 5.36997 9.49001 4.95997 9.32001 4.65997C7.94001 4.89997 6.63001 5.31997 5.40001 5.88997C2.92001 9.62997 2.25001 13.28 2.58001 16.87C4.23001 18.1 5.82001 18.84 7.39001 19.33C7.78001 18.8 8.12001 18.23 8.42001 17.64C7.85001 17.43 7.31001 17.16 6.80001 16.85C6.94001 16.75 7.07001 16.64 7.20001 16.54C10.33 18 13.72 18 16.81 16.54C16.94 16.65 17.07 16.75 17.21 16.85C16.7 17.16 16.15 17.42 15.59 17.64C15.89 18.23 16.23 18.8 16.62 19.33C18.19 18.84 19.79 18.1 21.43 16.87C21.82 12.7 20.76 9.08997 18.61 5.88997H18.59ZM8.84001 14.67C7.90001 14.67 7.13001 13.8 7.13001 12.73C7.13001 11.66 7.88001 10.79 8.84001 10.79C9.80001 10.79 10.56 11.66 10.55 12.73C10.55 13.79 9.80001 14.67 8.84001 14.67ZM15.15 14.67C14.21 14.67 13.44 13.8 13.44 12.73C13.44 11.66 14.19 10.79 15.15 10.79C16.11 10.79 16.87 11.66 16.86 12.73C16.86 13.79 16.11 14.67 15.15 14.67Z" fill="currentColor"/></svg>""",  # noqa
        },
        "github": {
            "link": "https://github.com/explodinggradients/ragas",
            "icon": (
                '<svg height="26px" style="margin-top:-2px;display:inline" '
                'viewBox="0 0 45 44" '
                'fill="currentColor" xmlns="http://www.w3.org/2000/svg">'
                '<path fill-rule="evenodd" clip-rule="evenodd" '
                'd="M22.477.927C10.485.927.76 10.65.76 22.647c0 9.596 6.223 17.736 '
                "14.853 20.608 1.087.2 1.483-.47 1.483-1.047 "
                "0-.516-.019-1.881-.03-3.693-6.04 "
                "1.312-7.315-2.912-7.315-2.912-.988-2.51-2.412-3.178-2.412-3.178-1.972-1.346.149-1.32.149-1.32 "  # noqa
                "2.18.154 3.327 2.24 3.327 2.24 1.937 3.318 5.084 2.36 6.321 "
                "1.803.197-1.403.759-2.36 "
                "1.379-2.903-4.823-.548-9.894-2.412-9.894-10.734 "
                "0-2.37.847-4.31 2.236-5.828-.224-.55-.969-2.759.214-5.748 0 0 "
                "1.822-.584 5.972 2.226 "
                "1.732-.482 3.59-.722 5.437-.732 1.845.01 3.703.25 5.437.732 "
                "4.147-2.81 5.967-2.226 "
                "5.967-2.226 1.185 2.99.44 5.198.217 5.748 1.392 1.517 2.232 3.457 "
                "2.232 5.828 0 "
                "8.344-5.078 10.18-9.916 10.717.779.67 1.474 1.996 1.474 4.021 0 "
                "2.904-.027 5.247-.027 "
                "5.96 0 .58.392 1.256 1.493 1.044C37.981 40.375 44.2 32.24 44.2 "
                '22.647c0-11.996-9.726-21.72-21.722-21.72" '
                'fill="currentColor"/></svg>'
            ),
        },
    },
)

html_theme_options = asdict(theme_options)

# -- Myst NB Config -------------------------------------------------
nb_execution_mode = "off"
nb_execute_in_temp = True
