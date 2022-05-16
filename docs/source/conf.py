# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
sys.path.insert(0, os.path.abspath("../../src/segmentador"))


# -- Project information -----------------------------------------------------

project = "Ulysses Segmenter"
copyright = "2022, Felipe Alves Siqueira"
author = "Felipe Alves Siqueira"

# The full version, including alpha/beta/rc tags
release = "v0.2.1-beta"


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named "sphinx.ext.*") or your custom
# ones.
extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinxcontrib.mermaid",
    "numpydoc",
    "autoapi.extension",
    "sphinx.ext.autosectionlabel",
]

autoapi_type = "python"
autoapi_dirs = ["../../src/segmentador"]
autoapi_options = [
    "show-module-summary",
    "show-inheritance",
    "special-members",
]

autoapi_ignore = [
    "*migrations*",
    "**/input_handlers*",
    "**/output_handlers*",
    "**/adapters*",
    "**/helpers*",
    "**/tasks/base*",
]
autoapi_add_toctree_entry = False
autoapi_generate_api_docs = True

autosectionlabel_prefix_document = True

myst_heading_anchors = 4

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []

source_suffix = {
    ".rst": "restructuredtext",
    ".txt": "markdown",
    ".md": "markdown",
}

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "sphinx_rtd_theme"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]

# -- Extra --
import re

with open("../../README.md", "r") as f_in_readme, open("../../LICENSE", "r") as f_in_license:
    new_readme = f_in_readme.read()
    new_readme = re.sub(r"\n##", "\n#", new_readme)
    new_readme = new_readme.replace("[MIT.](./LICENSE)", f"```markdown\n{f_in_license.read()}\n```\n")
    new_readme = new_readme.replace("```mermaid", "```{mermaid}\n  :align: center")

with open("./_static/PREPROCESSED_README.md", "w") as f_out_readme:
    f_out_readme.write(new_readme)
