import time

from progressbar import ProgressBar

from blessings import Terminal
from multiprocessing import Pool


term = Terminal()

location = (0, 10)
text = 'blessings!'
print(term.location(*location), text)

# alternately,
# with term.location(*self.location):
#     print(text)

class Writer(object):
    """Create an object with a write method that writes to a
    specific place on the screen, defined at instantiation.

    This is the glue between blessings and progressbar.
    """
    def __init__(self, location):
        """
        Input: location - tuple of ints (x, y), the position
                        of the bar in the terminal
        """
        self.location = location

    def write(self, string):
        with term.location(*self.location):
            print(string)

    def flush(self):
        pass

def test_function(location):
    writer = Writer(location)
    pbar = ProgressBar(fd=writer)
    # pbar.start()
    for i in range(100):
        # mimic doing some stuff
        time.sleep(0.01)
        pbar.update(i)
    pbar.finish()
locations = [(0, 1), (0, 6), (0, 7)]

p = Pool()
p.map(test_function, locations)
p.close()