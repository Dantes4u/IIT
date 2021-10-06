import colored
import time


class Logger:
    def _log(self, style, text):
        reset = colored.attr('reset')
        date = time.strftime("%Y-%m-%d %H:%M:%S")
        print(style + "[{:19s}] {}".format(date, text) + reset)

    def warning(self, text):
        self._log(colored.fg("dark_orange_3a"), text)

    def info(self, text):
        self._log(colored.fg("sky_blue_2"), text)

    def error(self, text):
        self._log(colored.fg("red_3a"), text)
