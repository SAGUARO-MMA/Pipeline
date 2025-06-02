import time
try:
    from StringIO import StringIO
except ImportError:
    from io import StringIO
from slack_sdk import WebClient
import logging
import os


class MyLogger(object):
    """
    Logger to control logging and uploading to slack.
    """

    def __init__(self, log, log_stream, slack_client):
        self._log = log
        self._log_stream = log_stream
        self.slack_client = slack_client

    def info(self, text):
        """
        Logs messages to log file at the INFO level.
        """
        self._log.info(text)

    def error(self, text):
        """
        Logs messages to log file at the ERROR level.
        """
        self._log.error(text)

    def critical(self, text):
        """
        Logs messages to log file at the CRITICAL level.
        """
        self._log.critical(text)
        try:
            self.slack_client.chat_postMessage(channel='CDY2K2F9V', text=text)
        except:  # if connection error occurs, add to log
            self._log.error('Connection error: failed to connect to slack. Above message not uploaded.')

    @staticmethod
    def shutdown():
        return logging.shutdown()


def initialize_logger(log_file_name):
    log_stream = StringIO()  # create log stream for upload to slack
    log = logging.getLogger(log_file_name)  # create logger
    log.setLevel(logging.INFO)  # set level of logger
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")  # set format of logger
    logging.Formatter.converter = time.gmtime  # convert time in logger to UTC
    filehandler = logging.FileHandler(log_file_name + '.log', 'w+')  # create log file
    filehandler.setFormatter(formatter)  # add format to log file
    log.addHandler(filehandler)  # link log file to logger
    streamhandler_slack = logging.StreamHandler(log_stream)  # add log stream to logger
    streamhandler_slack.setFormatter(formatter)  # add format to log stream
    log.addHandler(streamhandler_slack)  # link logger to log stream
    slack_client = WebClient(os.environ['SLACK_API_TOKEN'])
    return MyLogger(log, log_stream, slack_client)  # load logger handler
