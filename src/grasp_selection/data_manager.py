from abc import ABCMeta, abstractmethod

import datetime as dt
import experiment_config as ec
import IPython
import logging
import gce
import instance
import os
import signal
import sys
import time

MGR_DIR = '.dexnet'
MGR_PID_FILE = '.server.pid'
MGR_REQUEST_EXT = '.req'
MGR_RELEASE_EXT = '.rls'
MGR_SLEEP_TIME = 1
PUSH_QUEUE_FILE = 'push.que'
USERS_KEY = 'users'

REQUEST_SIGNAL = signal.SIGUSR1
RELEASE_SIGNAL = signal.SIGUSR2
TERM_SIGNAL = signal.SIGTERM

class DataManager:
    """
    Class for handling data write access requests (and possibly future read access requests)
    """

    def __init__(self, config, auto_start=True):
        """ Start the process if none exists. Otherwise create a handle to the true process """
        self.cache_dir_ = os.path.join(config['cache_dir'], MGR_DIR)
        self.running_ = False
        self.shutdown_ = False
        self.access_ = False
        self.config_ = config
        self.pid_ = os.getpid()
        self.server_pid_ = 0
        self.requests_ = []

        # setup signal handlers for requests and releases
        signal.signal(REQUEST_SIGNAL, self._handle_write_access_granted)
        signal.signal(RELEASE_SIGNAL, self._handle_write_access_released)

        # make sure data directory exists
        if not os.path.exists(self.cache_dir_):
            os.makedirs(self.cache_dir_)

        # start up server
        if auto_start:
            self.start()

    def _handle_write_access_granted(self, signum, frame):
        self.access_ = True

    def _handle_write_access_released(self, signum, frame):
        self.access_ = False

    def _handle_stop_request(self, signum, frame):
        """ Shutdown the standalone server """
        logging.info('Data manager received stop request')
        if len(self.requests_) == 0:
            self.shutdown_ = True
            os.remove(os.path.join(self.cache_dir_, MGR_PID_FILE))

    def _handle_force_stop_request(self, signum=0, frame=0):
        """ Shutdown the standalone server """
        logging.info('Data manager received stop request')
        self.shutdown_ = True
        
        # remove PID
        pid_filename = os.path.join(self.cache_dir_, MGR_PID_FILE)
        if os.path.exists(pid_filename):
            os.remove(pid_filename)

        # clear all remaining requests / releases
        remove_files = os.listdir(self.cache_dir_)
        for filename in remove_files:
            if filename.endswith(MGR_REQUEST_EXT) or filename.endswith(MGR_RELEASE_EXT):
                r_filename = os.path.join(self.cache_dir_, filename)
                if os.path.exists(r_filename):
                    os.remove(r_filename)

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

    def stage(self):
        """ Stages changes to the remote database. Not implemented for base class """
        return False

    def push(self):
        """ Pushes changes to the remote database. Not implemented for base class """
        return False

    def start(self):
        """ Start up the server """
        logging.info('Starting server')
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
            time.sleep(MGR_SLEEP_TIME)
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
        signal.signal(signal.SIGINT, self._handle_force_stop_request)

        # stage changes
        try:
            logging.info('Staging changes')
            self.stage()
        except Exception as e:
            logging.error('Staging failed')
            self._handle_force_stop_request()
            raise e

        # until all requests are finished, loop and wait for data write requests
        while not self.shutdown_:
            time.sleep(MGR_SLEEP_TIME)
            new_request = (len(self.requests_) == 0)
            released = False

            # check for new files
            # TODO: longer term, less hardcoding of file system calls
            candidate_files = self.get_request_candidates()
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

        # push changes
        try:
            logging.info('Pushing changes')
            self.push()
        except Exception as e:
            logging.error('Failed to push changes')
            self._handle_force_stop_request()
            raise e

        logging.info('Server shutting down...')
        exit(0)

    def stop(self):
        """ Request the manager server to stop. Only works if all requests are served """
        self.shutdown_ = True
        if self.running():
            os.kill(self.server_pid_, signal.SIGTERM)

    def force_stop(self):
        """ Force the manager server to stop """
        self.shutdown_ = True
        if self.running():
            os.kill(self.server_pid_, signal.SIGINT)

    def get_request_candidates(self):
        """ Returns the list of request candidates """
        return os.listdir(self.cache_dir_)  

    def request_write_access(self):
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
            time.sleep(MGR_SLEEP_TIME)
        return self.access_

    def release_write_access(self):
        """ For processes to release access to the data server """
        # submit release via touching a file
        release_file_name = os.path.join(self.cache_dir_, str(self.pid_) + MGR_RELEASE_EXT)
        f = open(release_file_name, 'w')
        f.write('%d\n' %(self.pid_))
        f.close()

        # wait for signal
        while self.access_:
            time.sleep(MGR_SLEEP_TIME)
        return not self.access_

class GceDataManager(DataManager):
    def __init__(self, config, auto_start=True):
        self.update_data_disks_ = []
        DataManager.__init__(self, config, auto_start=auto_start)

    @property
    def update_data_disks(self):
        if len(self.update_data_disks_) == 0:
            self.compute_update_data_disks()
        return self.update_data_disks_

    def compute_update_data_disks(self):
        """ Computes the names of update data disks """
        self.update_data_disks_ = []
        compute_config = self.config_['compute']

        # get update data disk names
        for zone, disk in zip(compute_config['zones'], compute_config['data_disks']):
            # create update disk names
            update_disk_name = '%s-update' %(disk)
            self.update_data_disks_.append(update_disk_name)        

    def stage(self):
        """ Stages changes by creating update disks, if they don't already exist. """
        # setup vars
        compute_config = self.config_['compute']
        created_snapshots = False
        if not self.update_data_disks_:
            self.compute_update_data_disks()

        # authorize access to GCE api
        auth_http = instance.oauth_authorization(self.config_)
        gce_helper = gce.Gce(auth_http, self.config_, project_id=compute_config['project'])

        # for all zones, create a disk snapshot if they don't already exist
        for zone, disk, update_disk_name in zip(compute_config['zones'], compute_config['data_disks'], self.update_data_disks_):
            # check for existence of the update disk (taken as a flag for the existence of an update node)
            disk_valid = gce_helper.get_disk(update_disk_name, zone)
            if not disk_valid:
                # create a snapshot of the current disk
                logging.info('Snapshotting disk %s' %(disk))                    
                snapshot_response = gce_helper.snapshot_disk(disk, compute_config['project'], zone)

                # create a disk from the snapshot
                logging.info('Creating update disk %s from snapshot %s' %(update_disk_name, snapshot_response['snapshot_name']))
                gce_helper.create_disk(update_disk_name, zone=zone, size_gb=compute_config['disk_size_gb'],
                                       source_snapshot=snapshot_response['snapshot_name'])

                # delete the snapshot
                ss_del_response = gce_helper.delete_snapshot(snapshot_name=snapshot_response['snapshot_name'], project=compute_config['project'])
                created_snapshots = True
        return created_snapshots

    def push(self):
        """ Pushed changes by replacing original disks with update disks. Super critical section. """
        # setup vars
        compute_config = self.config_['compute']
        dt_now = dt.datetime.now()
        if not self.update_data_disks_:
            self.compute_update_data_disks()

        # authorize access to GCE api
        auth_http = instance.oauth_authorization(self.config_)
        gce_helper = gce.Gce(auth_http, self.config_, project_id=compute_config['project'])

        for zone, disk, update_disk in zip(compute_config['zones'], compute_config['data_disks'], self.update_data_disks_):
            # check for update disk existence
            disk_response = gce_helper.get_disk(update_disk, zone)
            if not disk_response:
                logging.error('Update disk %s does not exist' %(update_disk))
                continue

            # generate backup disk filename
            backup_disk = '%s-backup-%s-%s-%s-%sh-%sm-%ss' %(disk, dt_now.month, dt_now.day, dt_now.year, dt_now.hour, dt_now.minute, dt_now.second) 

            # snapshot the updated data disks
            snapshot_response = gce_helper.snapshot_disk(update_disk, compute_config['project'], zone, name=backup_disk)

            # delete previous disk and replace, if not in use
            disk_response = gce_helper.get_disk(disk, zone)
            if disk_response:
                if USERS_KEY not in disk_response.keys() or (USERS_KEY in disk_response.keys() and len(disk_response[USERS_KEY]) == 0):
                    # create new disk from snapshot
                    gce_helper.delete_disk(disk, zone=zone)
                    gce_helper.create_disk(disk, zone=zone, size_gb=compute_config['disk_size_gb'],
                                           source_snapshot=snapshot_response['snapshot_name'])

                    # delete update disk (don't delete if push can't be done now, otherwise changes won't be overwritten)
                    gce_helper.delete_disk(update_disk, zone=zone)

                elif USERS_KEY in disk_response.keys() and len(disk_response[USERS_KEY]) > 0:
                    # stage the push for a future time
                    logging.info('Master disk %s in use. Staging backup disk for a future push' %(disk))
                    push_queue_filename = os.path.join(self.cache_dir_, PUSH_QUEUE_FILE)
                    f = open(push_queue_filename, 'a')
                    f.write(backup_disk + '\n')
            else:
                logging.warning('Master disk was not found') 

        return True

def push_update_disk(config):
    manager = GceDataManager(config, auto_start=False)
    manager.push()

def test_access(config):
    os.fork()
    os.fork()
    os.fork()

    manager = DataManager(config)

    manager.request_write_access()
    print 'Got access', manager.pid_
    time.sleep(2)
    manager.release_write_access()
    print 'Released access', manager.pid_

    time.sleep(2)
    manager.stop()    

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    argc = len(sys.argv)

    if argc < 2:
        logging.error('Must supply config file name')

    config_filename = sys.argv[1]
    config = ec.ExperimentConfig(config_filename)

    if argc == 2:
        mode = 'test'
    else:
        mode = sys.argv[2]

    if mode == 'test':
        test_access(config)
    elif mode == 'push':
        push_update_disk(config)
