# Carpenter

A repository of tools for working with Merian pixels. Note that this is *NOT* our basic package for catalog manipulation!

* Please read the guideline [here](https://github.com/mpi-astronomy/mpia-python-template) before contributing to this project.
* Remember to `pip install spherical-geometry` before using the package!

**Installation**
First of all, update your pip to the latest version.
```
python3 -m pip install --upgrade pip
```
With the package structure used here, you do not have to point Python to the location of your package. You absolutely SHOULD NOT be adding the package directory to your `$PYTHONPATH`. Instead, you can use `pip` to install it locally:
```
cd ~/path/to/new_project
pip install -e .
```
`pip` will install all the dependencies specified in the `setup.cfg` file. The `-e` flag makes the install editable which means that you do not have to install the package again and again after each change. Changes to your files inside the project folder will automatically reflect in changes on your installed package. If you are working in an interactive environment (`ipython`, `Jupyter`) you will need to re-import any modules that have changed. For example, after editing `module_x.py` you will need to do the following to have the changes available in the Python interpreter:

```
import importlib
importlib.reload(module_x)
```

To install a non-editable version, do:
```
cd ~/path/to/new_project
pip install .
```
This is how you can use your package once you are no longer developing it. Any users who are not contributing code can installing your package with:
```
pip install git+https://github.com/MerianSurvey/carpenter.git
```

**Commit early and often**

As you make changes to your package, get into the habit of committing changes early and often. Every time you add a new function, a new test, edit the docstring:
```
git add new_module.py
git commit -m "Added a function to reverse the sprocket of the whoosle."
```
And every few commits:
```
git push
```

**Testing your code**

Ideally, you should be writing tests along with the new code. To test your code, first install the test dependencies:
```
pip install -e ".[test]"
```

Then run the tests from the `new_project` directory:
```
pytest --cov=.
```

The `--cov=.` flag generates a report on how much of you code is covered by tests. Ideally this should be >80%.

To check for compliance with the Python style guide, run `flake8`:
```
flake8
```

This repository come pre-set with continuous integration using GitHub Actions. Every time you push a commit or make a pull request, all tests will be automatically run by GitHub. On the GitHub page for your repository you should have an `Action` tab (forth from the left). This tab will show you the test results. While you can (and should) run the test suite locally, these runs are usually only on your operating system against one version of python. The advantage of CI is that you can test your code against different versions of python, different versions of key libraries and different operating systems. Here we have set up a simple test matrix which runs against four different versions of python. You can make the CI more complex if you need. You can disable/enable actions as shown [here](https://docs.github.com/en/actions/managing-workflow-runs/disabling-and-enabling-a-workflow) or by deleting the `.github/workflows/ci.yml` file (use `git rm`). 

**Creating documentation**

If/when you want to update the auto-generated `sphinx` documentation, you can edit the `docs/index.rst` file. This file is in reStructuredText format. More information on making your docs pretty is available in the `sphinx` [docs](https://www.sphinx-doc.org/en/master/tutorial/index.html).

To generate the documentation, you need to first install the dependencies and then make the pages:
```
pip install -e ".[docs]"
cd docs/
make html
open _build/html/index.html
```
`sphinx` can also generate a PDF of your docs, but this is left as an exercise for the user.

This repository is also set to auto-generate an HTML page with the documentation and creates a GitHub pages webpage. While the files are auto-generated, the page must be made visible in the first place. Go to the `Settings` tab in GitHub and in the left-hand menu navigate to the `Pages` option. Select the `gh-pages` branch in the drop down `Source` menu. This is a one-time setting. The URL for your documentation will be displayed in the green banner. The example documentation page for this repository can be found at [https://meriansurvey.github.io/carpenter/](https://meriansurvey.github.io/carpenter/). 

You can disable/enable the auto-generated documentation builds as shown [here](https://docs.github.com/en/actions/managing-workflow-runs/disabling-and-enabling-a-workflow) or by deleting the `.github/workflows/docs.yml` file (use `git rm`). To unpublish the documentation page, you also need to delete the `gh-pages` branch, see instructions [here](https://docs.github.com/en/pages/getting-started-with-github-pages/unpublishing-a-github-pages-site#unpublishing-a-project-site).
