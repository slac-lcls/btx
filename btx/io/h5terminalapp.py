try:
    import h5py
    import curses
    import numpy as np
    import matplotlib.pyplot as plt
    import sys
    import time

    DATASET = h5py._hl.dataset.Dataset

except ModuleNotFoundError:
    print('Cannot run the terminal application. Missing dependencies.')

class H5TerminalApp:
    def __init__(self, path: str):
        self._path: str = path
        self.arr1: np.ndarray = np.zeros([1])
        self.arr2: np.ndarray = np.zeros([1])
        # Open h5 file
        try:
            self.h5 = h5py.File(path)
        except FileNotFoundError:
            print(f'Small data not found at path: {path}')
            return

    def run(self):
        # Curses init
        self.stdscr = curses.initscr()
        curses.noecho()
        curses.cbreak()
        self.stdscr.keypad(True)

        self._cols = curses.COLS
        self._rows = curses.LINES

        self.y_offset: int = int(self._rows*0.10)
        self.title_bar = curses.newwin(self.y_offset, self._cols, 0, 0)
        self.main_window = curses.newwin(self._rows - self.y_offset,
                                         self._cols,
                                         self.y_offset,
                                         0)

        # self.title_bar.box()
        self.main_window.box()

        name = self._path.split('/')[-1]
        self.title = f'Small Data Explorer - {name}'

        curses.start_color()
        curses.use_default_colors()

        # Color pair n with foreground f and background b (n, f, b)
        curses.init_pair(1, curses.COLOR_WHITE, curses.COLOR_BLUE)

        # Set background colors of the app
        self.main_window.bkgd(curses.color_pair(1))
        self.title_bar.bkgd(curses.color_pair(1))

        self.cursor = Cursor(y = 1,
                             x = 4,
                             ylim = [1, self._rows - self.y_offset - 2],
                             xlim = [0, self._cols - 1])

        # Cursor positions
        # self.cursor_x: int = 10
        # self.cursor_y: int = 10

        # Display initial text.
        self._update_title_bar()
        self._refresh(self.title_bar)

        self._update_main_window(None)
        self._refresh(self.main_window)

        # Main event loop
        self.main_loop()

    def main_loop(self):
        """! Main event loop.

        Loop condition assigns any keypress to a variable, and breaks if 'q'.
        Other keypresses will update screen display, otherwise continues to
        display the current text.
        """
        while ((keypress := self.main_window.getch()) != ord('q')):
            self._update_title_bar()
            self._refresh(self.title_bar)

            self._update_main_window(keypress)
            self._refresh(self.main_window)

    def _update_title_bar(self):
        curses.curs_set(0)
        x = (self._cols // 2) - (len(self.title) // 2)

        # Add title
        self.title_bar.addstr(1, x, self.title, curses.color_pair(1))

    def _update_main_window(self, keypress):
        curses.curs_set(1)
        keys: list = list(self.h5.keys())
        y: int = int(self._rows*0.15)
        for i in range(len(keys)):
            self.main_window.addstr(i + 1, 4, keys[i])
        self._update_cursor(keypress)

    def _update_cursor(self, keypress):
        if keypress == ord('d'):
            # self.cursor.right()
            pass

        elif keypress == ord('a'):
            # self.cursor.left()
            pass

        elif keypress == ord('s'):
            self.cursor.down()

        elif keypress == ord('w'):
            self.cursor.up()

        self.main_window.move(self.cursor.row, self.cursor.col)
        curses.doupdate()

    def _refresh(self, window):
        window.timeout(1)
        window.refresh()

    def _cleanup(self, window):
        window.clear()
        window.refresh()

    def __exit__(self, *errors):
        self._cleanup(self.title_bar)
        self._cleanup(self.main_window )
        curses.nocbreak()
        self.stdscr.keypad(False)
        curses.echo()
        curses.endwin()

    def __del__(self):
        self.__exit__()


class Cursor:
    """! Holds the cursor position and limits on where it can move."""
    def __init__(self, y, x, ylim, xlim):
        self.row = y
        self.col = x

        self.ylim = ylim
        self.xlim = xlim

    def left(self):
        """! Move cursor position to the left by one."""
        if self.col > self.xlim[0]:
            self.col -= 1

    def right(self):
        """! Move cursor position to the right by one."""
        if self.col < self.xlim[1]:
            self.col += 1

    def up(self):
        """! Move cursor position up by one."""
        if self.row > self.ylim[0]:
            self.row -= 1

    def down(self):
        """! Move cursor position down by one."""
        if self.row < self.ylim[1]:
            self.row += 1

    def update_limits(self, ylim, xlim):
        """! Update the bounds of permitted x, y cursor positions."""
        self.ylim = ylim
        self.xlim = xlim

if __name__ == '__main__':
    app = H5TerminalApp(sys.argv[1])
    app.run()

