import logging
import tkinter as tk
import sys

class GuiHandler(logging.StreamHandler):
    '''
    Extention of a logging.StreamHandler to print message on a 
    tkinter Text widget

    HISTORY
    -------
    2020 11 05, Stefano Rapisarda (Uppsala), creation date
        I made a modified version of StreamHandler adding few
        lines to the native emit method. Now logging messages
        can also be printed on a tkinter text widget! Nice
    '''

    def __init__(self, text_widget=None, *args, **kwargs):
        logging.StreamHandler.__init__(self,*args,**kwargs)

        self._text_widget = text_widget

    def emit(self, record):
        """
        Emit a record.
        If a formatter is specified, it is used to format the record.
        The record is then written to the stream with a trailing newline.  If
        exception information is present, it is formatted using
        traceback.print_exception and appended to the stream.  If the stream
        has an 'encoding' attribute, it is used to determine how to do the
        output to the stream.
        """
        try:
            msg = self.format(record)
            stream = self.stream
            # issue 35046: merged two stream.writes into one.
            stream.write(msg + self.terminator)
            self.flush()

            # Added by Stefano
            if not self._text_widget is None:
                self._text_widget.insert(tk.END, msg+'\n')
        except RecursionError:  # See issue 36272
            raise
        except Exception:
            self.handleError(record)