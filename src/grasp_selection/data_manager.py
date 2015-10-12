from abc import ABCMeta, abstractmethod
import experiment_config as ec
import logging
import os
import signal
import sys
import time

MGR_DIR = '.dexnet'
MGR_PID_FILE = '.server.pid'
MGR_REQUEST_EXT = '.req'
MGR_RELEASE_EXT = '.rls'

REQUEST_SIGNAL = signal.SIGUSR1
RELEASE_SIGNAL = signal.SIGUSR2
TERM_SIGNAL = signal.SIGTERM

class DataManager:
    """
    Class for handling data write access requests (and possibly future read access requests)
    """

    def __init__(self, config):
        """ Start the process if none exists. Otherwise create a handle to the true process """
        self.cache_dir_ = os.path.join(config['cache_dir'], MGR_DIR)
        self.running_ = False
        self.shutdown_ = False
        self.access_ = False
        self.pid_ = os.getpid()
        self.server_pid_ = 0
        self.requests_ = []

        # setup signal handlers for requests and releases
        signal.signal(REQUEST_SIGNAL, self._handle_data_access_granted)
        signal.signal(RELEASE_SIGNAL, self._handle_data_access_released)

        # make sure data directory exists
        if not os.path.exists(self.cache_dir_):
            os.makedirs(self.cache_dir_)

        # start up server
        self.start()

    def _handle_data_access_granted(self, signum, frame):
        self.access_ = True

    def _handle_data_access_released(self, signum, frame):
        self.access_ = False

    def _handle_stop_request(self, signum, frame):
        """ Shutdown the standalone server """
        logging.info('Data manager received stop request')
        if len(self.requests_) == 0:
            self.shutdown_ = True
            os.remove(os.path.join(self.cache_dir_, MGR_PID_FILE))

    def running(self):
        """ Check if an instance of the manager is already running """
        # search for a PID file for the manager
        candidate_files = os.listdir(self.cache_dir_)
        for file_name in candidate_files:
            if file_name == MGR_PID_FILE:
                file_name = os.path.join(self.cache_dir_, file_name)

                try:
                    f = open(file_name, 'r')
                    self.server_pid_ = int(f.readline())
                except ValueError:
                    logging.warning('Server PID file accessed while in creation')
                    return False
                return True

        return False

    def start(self):
        """ Start up the server """
        # check for running server
        if self.running():
            return False

        # check for creation in the meantime
        file_name = os.path.join(self.cache_dir_, MGR_PID_FILE)
        if os.path.exists(file_name):
            return

        # launch child process
        f = open(file_name, 'w')
        self.server_pid_ = os.fork()
        if self.server_pid_ > 0: # parent process
            # create pid file
            f.write('%d\n' %(self.server_pid_))
        else:
            time.sleep(1)
            if not self.running():
                logging.error('Server not started. PID file did not exist')
                raise ValueError()
            self.pid_ = self.server_pid_
            logging.info('Server started with pid %d' %(self.pid_))
            self.run()

    def run(self):
        self.shutdown_ = False

        # setup server-specific signal handlers
        signal.signal(signal.SIGTERM, self._handle_stop_request)

        while not self.shutdown_:
            time.sleep(1)
            new_request = (len(self.requests_) == 0)
            released = False

            # check for new files
            candidate_files = os.listdir(self.cache_dir_)
            for file_name in candidate_files:
                # log access request
                if file_name.endswith(MGR_REQUEST_EXT):
                    file_name = os.path.join(self.cache_dir_, file_name)
                    f = open(file_name, 'r')
                    req_pid = int(f.readline())
                    logging.info('Server request from %d' %(req_pid))
                    os.remove(file_name) # remove request
                    self.requests_.append(req_pid)
                # log release request
                elif file_name.endswith(MGR_RELEASE_EXT):
                    file_name = os.path.join(self.cache_dir_, file_name)
                    f = open(file_name, 'r')
                    rel_pid = int(f.readline())
                    logging.info('Server release from %d' %(rel_pid))
                    os.remove(file_name) # remove release

                    # grant release if consistent with list
                    if rel_pid == self.requests_[0]:
                        os.kill(rel_pid, RELEASE_SIGNAL)
                        self.requests_.pop(0)
                        released = True

            # if release was given, give it to the next process
            if released or (new_request and len(self.requests_) > 0):
                if len(self.requests_) > 0:
                    logging.info('Granting access to %d' %(self.requests_[0]))
                    os.kill(self.requests_[0], REQUEST_SIGNAL)

        logging.info('Server shutting down...')
        exit(0)

    def stop(self):
        """ Stop the manager server """
        self.shutdown_ = True
        if self.running():
            os.kill(self.server_pid_, signal.SIGTERM)

    def request_data_access(self):
        """ Blocking access request to the data server """
        self.access_ = False

        # double-check that the server is running
        if not self.running():
            self.start()

        # submit request via touching a file
        request_file_name = os.path.join(self.cache_dir_, str(self.pid_) + MGR_REQUEST_EXT)
        f = open(request_file_name, 'w')
        f.write('%d\n' %(self.pid_))
        f.close()

        # wait for signal
        while not self.access_:
            time.sleep(1)
        return self.access_

    def release_data_access(self):
        """ For processes to release access to the data server """
        # submit release via touching a file
        release_file_name = os.path.join(self.cache_dir_, str(self.pid_) + MGR_RELEASE_EXT)
        f = open(release_file_name, 'w')
        f.write('%d\n' %(self.pid_))
        f.close()

        # wait for signal
        while self.access_:
            time.sleep(1)
        return not self.access_

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    argc = len(sys.argv)

    if argc < 2:
        logging.error('Must supply config file name')

    config_filename = sys.argv[1]
    config = ec.ExperimentConfig(config_filename)

    os.fork()
    os.fork()
    os.fork()

    manager = DataManager(config)

    manager.request_data_access()
    print 'Got access', manager.pid_
    time.sleep(2)
    manager.release_data_access()
    print 'Released access', manager.pid_

    time.sleep(2)
    manager.stop()
