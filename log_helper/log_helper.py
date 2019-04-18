import os
import sys
import logging
import pathlib
from datetime import datetime
from pytz import timezone, utc


class LogHelper:
    log_format = '%(asctime)s %(levelname)s - %(name)s - %(message)s'

    @staticmethod
    def setup(log_path, log_level=logging.INFO):
        # TODO: Add exception handling if parent directory of log_path does not exist
        logging.basicConfig(
             filename=log_path,
             level=log_level, 
             format= LogHelper.log_format,
         )

        def customTime(*args):
            utc_dt = utc.localize(datetime.utcnow())
            my_tz = timezone("Asia/Hong_Kong")
            converted = utc_dt.astimezone(my_tz)
            return converted.timetuple()

        logging.Formatter.converter = customTime

        # Add the console handler to the root logger
        logger = logging.getLogger(None)
        # avoid adding handlers multiple times
        if len(logger.handlers) < 2:
            # Set up logging to console
            console = logging.StreamHandler()
            console.setLevel(logging.DEBUG)
            console.setFormatter(logging.Formatter(LogHelper.log_format))
            logger.addHandler(console)

        # Log for unhandled exception
        logger = logging.getLogger(__name__)
        sys.excepthook = lambda *ex: logger.critical('Unhandled exception', exc_info=ex)

        logger.info('Finished configuring logger')
