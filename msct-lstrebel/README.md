# msct-lstrebel
Lukas Strebel Master Thesis Repository
======================================

Making the gridtools module available:
--------------------------------------

Only the first time, create a virtual environment:

```shell
    $ mkdir gtenv

    $ python3 -m venv gtenv/
```

Activate the virtual environment to work with the gridtools module.

```shell
    $ source gtenv/bin/activate
```

Then only the first time, install the requirements from the text file in the repository root:

```shell
    (gtenv) $ pip install -r requirements.txt
```

To make sure Python can find the gridtools module export the path to the PYTHONPATH environment variable every time:

```shell
    export PYTHONPATH=<Path-To-the-repository-root-folder>$PYTHONPATH
```

To make PyMetis now where the Metis library is export the following every time:


```shell
    export METIS_DLL=<Path-To-the-repository-root-folder>msct-lstrebel/external_libraries/metis-5.1.0/install/lib/libmetis.so
```



