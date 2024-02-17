# Installation

To get started, install Ragas using `pip` with the following command:

```bash
pip install ragas
```

If you'd like to experiment with the latest features, install the most recent version from the main branch:

```bash
pip install git+https://github.com/explodinggradients/ragas.git
```

If you're planning to contribute and make modifications to the code, ensure that you clone the repository and set it up as an [editable install](https://pip.pypa.io/en/stable/topics/local-project-installs/#editable-installs).

```bash
git clone https://github.com/explodinggradients/ragas.git 
cd ragas 
pip install -e .
```

Next, let's construct a [synthetic test set](get-started-testset-generation) using your own data. If you've brought your own test set, you can learn how to [evaluate it](get-started-evaluation) using Ragas.