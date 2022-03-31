import os
from pathlib import Path


def run_usage(usages):
    usages = [f for f in usages]

    print(usages)

    for path_to_file in usages:
        fname = os.path.basename(path_to_file)

        print(fname)

        with open(path_to_file) as f:
            s = f.read()

            no_plots = "import matplotlib\n" \
                       "import matplotlib.pyplot\n" \
                       "matplotlib.use('Agg')\n"

            s = no_plots + s + "\nmatplotlib.pyplot.close()\n"

            try:
                exec(s, globals())
            except:
                raise Exception("Usage %s failed." % fname)



def test_usages():
    folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "ezmodel", "usage")
    run_usage([os.path.join(folder, fname) for fname in Path(folder).glob('**/*.py')])

