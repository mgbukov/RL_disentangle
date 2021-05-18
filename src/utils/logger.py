import os


class FileLogger:
    def __init__(self, log_dir, filename):
        """ Initialize file logger object.

        @param log_dir (str): Directory for logging outputs.
        @param filename (str): Filename for logging to file.
        """
        self._log_dir = log_dir
        self._filename = filename
        self._verbose = False
        self.flush()


    def logTxt(self, s):
        """ Log a text string @s to a file. """
        with open(os.path.join(self._log_dir, self._filename), "a") as f:
            f.write(s)
            f.write("\n")
            if self._verbose:
                print(s)


    def flush(self):
        """ Delete the contents of the file, or create a new file if it doesn't exist. """
        f = open(os.path.join(self._log_dir, self._filename), "w")
        f.close()


    def verboseLogging(self, verbose):
        self._verbose = verbose

#