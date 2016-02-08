from abc import ABCMeta, abstractmethod

import IPython
import traceback

from apiclient import discovery
import copy
import data_manager as dm
import experiment_config as ec
import gce
import gcs
import instance
import logging
import multiprocessing as mp
import numpy as np
import os
import sys
import time

from oauth2client.client import AccessTokenRefreshError
import smtplib
import signal

EMAIL_NOTIFICATION = """
Your job %(instance_id)s has completed.

Here are the instances that were created:
%(instance_names)s

Here was the config used to run the experiment:

    %(experiment_config)s

Here is the set of scripts run:

    %(script_commands)s

Here's the output of `gcloud compute instances list`:

%(listinstances_output)s
"""

def gen_job_id(n=10):
    """ Random string for naming """
    chrs = 'abcdefghijklmnopqrstuvwxyz'
    inds = np.random.randint(0,len(chrs), size=n)
    return ''.join([chrs[i] for i in inds])

def wait_for_input(timeout, prompt='> ', no_input_msg=''):
    def handler(signum, frame):
        raise RuntimeError
    signal.signal(signal.SIGALRM, handler) # not portable...
    signal.alarm(timeout)
    try:
        text = raw_input(prompt)
        signal.alarm(0)
        return text
    except RuntimeError:
        print no_input_msg
    signal.signal(signal.SIGALRM, signal.SIG_IGN)
    return ''

class Job(object):
    __metaclass__ = ABCMeta

    @abstractmethod
    def _setup(self):
        """
        Partitions data and parameters for a set of instances. Typically would be a private function that is called from "init"
        """
        pass

    @abstractmethod
    def is_complete(self):
        """
        Returns true if the job is complete, false otherwise
        """
        pass

    @abstractmethod
    def spin(self):
        """
        Does any computation necessary to check completion, waits for next completion check, etc
        """
        pass

    @abstractmethod
    def start(self):
        """
        Launch all instances and stuff
        """
        pass

    @abstractmethod
    def stop(self):
        """
        Stops all instances and cleans up
        """
        pass

    def run(self):
        """
        Starts and stops a new experiment
        """
        start_time = time.time()
        self.start()
        while not self.is_complete():
            self.spin()
        self.stop()
        end_time = time.time()
        logging.info('Job took %f sec' %(end_time - start_time))

    @abstractmethod
    def analyze(self):
        """
        Analyzes the output of a job. For example, this can generate plots or compute error statistics
        """
        pass

    @abstractmethod
    def store(self):
        """
        Updates the database with any output from the job. This function is executed remotely on instances with read / write disk access.
        This should only be called by instances with write priveleges.
        """
        pass

    def send_notification_email(self, message, config, subject="Your job has completed."):
        """ Sends a notification email to the user specified in the config """
        # http://stackoverflow.com/questions/10147455/trying-to-send-email-gmail-as-mail-provider-using-python
        gmail_user = config['gmail_user']
        gmail_pwd = config['gmail_password']
        notify_email = config['notify_email']

        from_email = gmail_user + "@gmail.com"
        to_emails = [notify_email] #must be a list

        # Prepare actual message
        message = "From: %s\nTo: %s\nSubject: %s\n\n%s\n" % (from_email, ", ".join(to_emails), subject, message)
        server = smtplib.SMTP("smtp.gmail.com", 587) #or port 465 doesn't seem to work!
        server.ehlo()
        server.starttls()
        server.login(gmail_user, gmail_pwd)
        server.sendmail(from_email, to_emails, message)
        server.close()
        logging.info('successfully sent the mail')

class GceJob(Job):
    """ A Job to run on Google Compute Engine """
    def __init__(self, config):
        self.config = config
        self._setup()

    def _setup(self):
        # get job id
        self.id = gen_job_id()
        self.job_name_root = 'job-%s' %(self.id)
        if self.config['update']:
            self.job_name_root = 'job-updater-%s' %(self.id)            
        self.config['job_root'] = self.job_name_root

        # setup logging for job
        gce_job_log = os.path.join(self.config['log_dir'], self.job_name_root +'.log')
        hdlr = logging.FileHandler(gce_job_log)
        formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
        hdlr.setFormatter(formatter)
        logging.getLogger().addHandler(hdlr) 

        # initialize helper classes for interacting with GCE, GCS
        self.auth_http = instance.oauth_authorization(self.config)
        self.gce_helper = gce.Gce(self.auth_http, self.config, project_id=self.config['compute']['project'])
        self.gcs_helper = gcs.Gcs(self.auth_http, self.config, project_id=self.config['compute']['project'])
        self.instance_manager = instance.VMInstanceManager()

        self.instances = []
        self.launched_instances = {}
        self.user_terminated = False

        # setup instance completion api call
        service_not_ready = True
        while service_not_ready:
            try:
                service = discovery.build('storage', self.config['compute']['api_version'], http=self.gce_helper.auth_http)
                self.bucket_req = service.objects().list(bucket=self.config['compute']['bucket'], prefix=self.job_name_root)
                service_not_ready = False
            except (ValueError, Exception) as e:
                logging.info('Connection failed. Retrying...')

    def instances_in_progress(self):
        """ Returns a list of the instances still running """
        inst = []
        for instance in self.launched_instances.values():
            if instance.running:
                inst.append(instance)
        return inst

    def completed_instances(self):
        """ Returns a list of the instances still running """
        inst = []
        for instance in self.launched_instances.values():
            if not instance.running:
                inst.append(instance)
        return inst

    def is_complete(self):
        """ Complete if all flags checked off or the user terminated the job """
        return len(self.instances_in_progress()) == 0 or self.user_terminated

    def spin(self):
        # get user input
        self.user_terminated = wait_for_input(self.config['sleep_time'], prompt='done? ')
        
        # check for finished instances
        logging.info('Checking for completion...')
        try:
            resp = self.bucket_req.execute()
        except (ValueError, Exception) as e:
            logging.info('Connection failed. Retrying...')
            return
        try:
            items = resp['items']
        except KeyError as e:
            logging.error(e)
            logging.error(resp)
            return

        # grab out the instances in progress
        instances_to_stop = []
        for item in items:
            # get instance name root
            instance_name, file_ext = os.path.splitext(item['name'])
            instance_name, file_ext = os.path.splitext(instance_name)
            instance_names_in_progress = [instance.instance_name for instance in self.instances_in_progress()]

            if instance_name in instance_names_in_progress:
                # queue instance for termination
                instances_to_stop.append(self.launched_instances[instance_name])
                logging.info('Instance %s completed!' % instance_name)
        
        # terminate instances
        self.instance_manager.stop_instances(instances_to_stop)
        instance_names_in_progress = [instance.instance_name for instance in self.instances_in_progress()]
        logging.info('Instances in progress: %s', ' '.join(instance_names_in_progress))
        
    def start(self):
        """ Start the job! """
        self.user_terminated = False
        self.complete_instances = []
        try:
            # allocate instance list
            data_disks = [gce.Disk(name, mode) for name, mode in zip(self.config['compute']['data_disks'], self.config['compute']['data_disk_modes'])]
            gce_allocator = instance.GceInstanceAllocator(self.config, self.gce_helper)
            instances = gce_allocator.allocate(self.job_name_root, self.config['compute']['run_script'], data_disks)
 
            # launch instances using multiprocessing
            self.launched_instances = self.instance_manager.launch_instances(instances, self.config['num_processes'])
        except Exception as e:
            logging.error('Failed to launch instances')
            logging.error('Traceback: %s' %(traceback.print_exc()))
            return False
        return True

    def stop(self):
        """ Stop the job! """
        # stop any remaining instances
        self.instance_manager.stop_instances(self.instances_in_progress(), self.config['num_processes'])
        
        # list running instances
        all_running_instances = []
        for zone in self.config['compute']['zones']:
            zone_instances = self.gce_helper.list_instances(zone)
            lines = ['These are your running instances in zone %s:' %(zone)]
            for zone_instance in zone_instances:
                lines.append('    ' + zone_instance['name'])
            if not zone_instances:
                lines.append('    (none)')
            zone_instances_text = '\n'.join(lines)
            all_running_instances.append(zone_instances_text)

        # download experiment output
        completed_instance_results = [n.instance_name + '.tar.gz' for n in self.completed_instances()] 
        self.job_store_dir, _ = self.gcs_helper.retrieve_results(self.config['compute']['bucket'], completed_instance_results, self.job_name_root)

        # save config file to directory
        with open(os.path.join(self.job_store_dir, 'config.yaml'), 'w') as f:
            f.write(self.config.file_contents)

        # send notification email
        if self.config['notify_email'] is not None:
            message = EMAIL_NOTIFICATION % dict(
                instance_id=self.job_name_root,
                instance_names='\n'.join(map(lambda n: '    ' + n, self.launched_instances.keys())),
                experiment_config=self.config.filename,
                script_commands=self.config['compute']['startup_script'],
                listinstances_output='\n\n'.join(all_running_instances)
                )
            
            self.send_notification_email(message=message, config=self.config,
                                         subject="Your experiment has completed.")

    def analyze(self):
        """ Analyze the results """
        # just call an analysis script for now
        if self.config['results_script']:
            start_time = time.time()
            results_script_call = 'python %s %s %s' %(self.config['results_script'], self.config.filename, self.job_store_dir)
            end_time = time.time()
            logging.info('Result analysis took %f sec' %(end_time - start_time))

    def store(self, wait_time=1.0):
        """
        Launch a set of update disks in read / write mode, then update, then replace old disks.
        Waits until update disks are free, launches a job with the update script, then replicates the disks if not in use
        """
        # create update configurations
        update_config = copy.copy(self.config)
        update_config['use_hard_limits'] = False
        update_config['update'] = True

        compute_config = self.config['compute']

        # create data disk names, ensure their availability
        if compute_config['update_script'] is not None:
            # create a data manager
            data_manager = dm.GceDataManager(self.config)
            try:
                # get write access
                data_manager.request_write_access()

                # now launch one instance per disk to do the updating
                update_config['compute']['data_disks'] = data_manager.update_data_disks
                update_config['compute']['data_disk_modes'] = ['READ_WRITE' for a in data_manager.update_data_disks]
                update_config['compute']['instance_quota'] = 1
                update_config['compute']['run_script'] = compute_config['update_script']
                update_config['prompt'] = 0 # turn off prompting
                logging.info('Running update job with script %s' %(compute_config['update_script']))

                update_job = GceJob(update_config)
                update_config['job_root'] = self.job_name_root
                
                update_job.run()

                # release access and attempt to stop the server
                data_manager.release_write_access()
                data_manager.stop()
                return True
            except Exception as e:
                # just make sure to terminate the goddamn server
                data_manager.force_stop()
                raise e
        return False

def test_gce_job_run():
    config_name = 'cfg/test_gce.yaml'
    config = ec.ExperimentConfig(config_name)
    gce_job = GceJob(config)
    gce_job.run()

def test_gce_job_update():
    config_name = 'cfg/test_gce_update2.yaml'
    config = ec.ExperimentConfig(config_name)
    gce_job = GceJob(config)
    gce_job.store()

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    test_gce_job_update()
