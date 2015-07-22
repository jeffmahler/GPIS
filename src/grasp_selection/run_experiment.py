# Copyright 2013 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
DexNet version of the Compute Engine demo, main.py.
Launches an instance with a random name to run the experiment configured in the input yaml file
See below for original file description:

Google Compute Engine demo using the Google Python Client Library.
__author__ = 'kbrisbin@google.com (Kathryn Hurley)'

Demo steps:
- Create an instance with a start up script and metadata.
- Print out the URL where the modified image will be written.
- The start up script executes these steps on the instance:
    - Installs Image Magick on the machine.
    - Downloads the image from the URL provided in the metadata.
    - Adds the text provided in the metadata to the image.
    - Copies the edited image to Cloud Storage.
- After recieving input from the user, shut down the instance.

To run this demo:
- Edit the client id and secret in the client_secrets.json file.
- Enter your Compute Engine API console project name below.
- Enter the URL of an image in the code below.
- Create a bucket on Google Cloud Storage accessible by your console project:
http://cloud.google.com/products/cloud-storage.html
- Enter the name of the bucket below.
"""

__author__ = 'jmahler@berkeley.edu (Jeff Mahler)'

import argparse
import IPython
import logging
try:
    import simplejson as json
except:
    import json
import numpy as np
import os
import shutil
import sys
import time

import multiprocessing as mp

from apiclient import discovery
import gce
import gcs
import httplib2
from oauth2client.client import flow_from_clientsecrets
from oauth2client.file import Storage
from oauth2client.tools import argparser, run_flow
from oauth2client import tools
import smtplib

import experiment_config as ec

INSTANCE_NAME_LENGTH = 10

INSERT_ERROR = 'Error inserting %(name)s.'
DELETE_ERROR = """
Error deleting %(name)s. %(name)s might still exist; You can use
the console (http://cloud.google.com/console) to delete %(name)s.
"""
EMAIL_NOTIFICATION = """
Your experiment %(instance_id)s has completed.

Here are the instances that were created:
%(instance_names)s

Here was the config used to run the experiment:

    %(experiment_config)s

Here is the set of scripts run:

    %(script_commands)s

Here's the output of `gcloud compute instances list`:

%(listinstances_output)s
"""

# global queue variable (stupid, but need to just get it running)
QUEUE_TIMEOUT = 10
MAX_QUEUE_SIZE = 1000
instance_launch_queue = mp.Queue(MAX_QUEUE_SIZE)

def launch_instance(instance):
    """ Launches an instance object """
#    instance.start()
    try:
        if instance.start():
            instance_launch_queue.put(instance.instance_name, timeout=QUEUE_TIMEOUT)
    except:
        logging.info('Failed to launch %s' %(instance.instance_name))

def stop_instance(instance):
    """ Stops an instance object """
    instance.stop()

class GceInstance:
    def __init__(self, instance_name, disk_name, image_name, zone, metadata, config):
        self.instance_name = instance_name
        self.disk_name = disk_name
        self.image_name = image_name
        self.zone = zone
        self.metadata = metadata
        self.project = config['project']
        self.config = config

    def create_gce_helper(self):
        """ Create a gce helper class configured and authenticated by this object """
        # authorize local process
        auth_http = oauth_authorization(self.config, None)

        # helper for gce api calls
        gce_helper = gce.Gce(auth_http, self.config, self.project)
        return gce_helper

    def delete_disk(self):
        """ Attempt to delete the instance disk """
        try:
            logging.info('Deleting %s' % self.disk_name)
            gce_helper = self.create_gce_helper()
            gce_helper.delete_disk(self.disk_name, self.zone)
        except (gce.ApiError, gce.ApiOperationError, ValueError) as e:
            logging.error(DELETE_ERROR, {'name': self.disk_name})
            logging.error(e)
            return False
        return True

    def terminate_instance(self):
        """ Attempt to terminate the running instance """
        try:
            logging.info('Stopping %s' % self.instance_name)
            gce_helper = self.create_gce_helper()
            gce_helper.stop_instance(self.instance_name, self.zone)
        except (gce.ApiError, gce.ApiOperationError, ValueError) as e:
            logging.error(DELETE_ERROR, {'name': self.instance_name})
            logging.error(e)
            return False
        return True

    def start(self):
        """ Launch a gce instance """
        logging.info('Starting GCE instance %s' % self.instance_name)
        try:
            # create gce and launch
            gce_helper = self.create_gce_helper()
            gce_helper.start_instance(
                self.instance_name, self.disk_name, self.image_name,
                service_email = self.config['compute']['service_email'],
                scopes = self.config['compute']['scopes'],
                startup_script = self.config['compute']['startup_script'],
                zone = self.zone,
                metadata = self.metadata,
                additional_ro_disks = self.config['compute']['data_disks']
            )
        except (gce.ApiError, gce.ApiOperationError, ValueError, Exception) as e:
            # Delete the disk in case the instance fails to start.
            self.delete_disk()
            logging.error(INSERT_ERROR, {'name': self.instance_name})
            logging.error(e)
            return False
        except gce.DiskDoesNotExistError as e:
            logging.error(INSERT_ERROR, {'name': self.instance_name})
            logging.error(e)
            return False

        instance_console = ('https://console.developers.google.com/'
                            'project/nth-clone-620/compute/instancesDetail/'
                            'zones/us-central1-a/instances/%s/console#end') % self.instance_name
        logging.info('Instance %s is running! Check it out: %s' %(self.instance_name, instance_console))        
        return True

    def stop(self):
        """ Stop and cleanup this instance """
        # stop instance
        self.terminate_instance()
        # delete disk
        self.delete_disk()

def send_notification_email(message, config, subject="Your experiment has completed."):
    # http://stackoverflow.com/questions/10147455/trying-to-send-email-gmail-as-mail-provider-using-python
    gmail_user = config['gmail_user']
    gmail_pwd = config['gmail_password']
    notify_email = config['notify_email']

    from_email = gmail_user + "@gmail.com"
    to_emails = [notify_email] #must be a list

    # Prepare actual message
    message = "From: %s\nTo: %s\nSubject: %s\n\n%s\n" % (from_email, ", ".join(to_emails), subject, message)
    #server = smtplib.SMTP(SERVER)
    server = smtplib.SMTP("smtp.gmail.com", 587) #or port 465 doesn't seem to work!
    server.ehlo()
    server.starttls()
    server.login(gmail_user, gmail_pwd)
    server.sendmail(from_email, to_emails, message)
    #server.quit()
    server.close()
    logging.info('successfully sent the mail')

def random_string(n):
    """
    Random string for naming
    """
    chrs = 'abcdefghijklmnopqrstuvwxyz'
    inds = np.random.randint(0,len(chrs), size=n)
    return ''.join([chrs[i] for i in inds])

def make_chunks(config):
    """Chunks datasets according to configuration. Each chunk only contains
    data from one dataset."""
    # Read counts file
    counts = {}
    with open(config['dataset_counts']) as f:
        for line in f:
            count, dataset = line.split()
            counts[dataset] = int(count)

    # Create chunks
    datasets = config['datasets']
    max_chunk_size = config['max_chunk_size']
    chunks = []
    for dataset in datasets:
        assigned = 0
        while assigned < counts[dataset]:
            chunk = dict(dataset=dataset,
                         chunk=[assigned, assigned+max_chunk_size])
            chunks.append(chunk)
            assigned += max_chunk_size
    yesno = raw_input('Create %d instances? [Y/n] ' % len(chunks))
    if yesno.lower() == 'n':
        sys.exit(1)
    return chunks

def oauth_authorization(config, args):
    """
    Perform OAuth2 authorization and return an authorized instance of
    httplib2.Http.
    """
    # Perform OAuth 2.0 authorization flow.
    flow = flow_from_clientsecrets(
        config['client_secrets'], scope=config['compute']['scopes'])
    storage = Storage(config['oauth_storage'])
    credentials = storage.get()

    # Authorize an instance of httplib2.Http.
    logging.info('Authorizing Google API')
    if credentials is None or credentials.invalid:
        credentials = run_flow(flow, storage, args)
    http = httplib2.Http()
    auth_http = credentials.authorize(http)
    return auth_http


def launch_experiment(args, sleep_time):
    """
    Perform OAuth 2 authorization, then start, list, and stop instance(s).
    """
    # Get total runtime
    start_time = time.time()
    launch_prep_start_time = time.time()

    # Parse arguments and load config file.
    config_file = args.config
    config = ec.ExperimentConfig(config_file)
    logging.basicConfig(level=logging.INFO)
    auth_http = oauth_authorization(config, args)

    # Retrieve / create instance data
    bucket = config['bucket']
    if not bucket:
        logging.error('Cloud Storage bucket required.')
        return
    instance_id = random_string(INSTANCE_NAME_LENGTH)
    instance_root = 'experiment-%s' %(instance_id)
    instance_name = '%s-' %(instance_root) + '%d'
    disk_name = instance_name + '-disk'
    image_name = config['compute']['image']
    run_script = config['compute']['run_script']

    # Make chunks
    chunks = make_chunks(config)

    # Initialize gce.Gce
    logging.info('Initializing GCE')
    gce_helper = gce.Gce(auth_http, config, project_id=config['project'])
    gcs_helper = gcs.Gcs(auth_http, config, project_id=config['project'])

    # Start an instance for each chunk
    num_instances = 0
    instances_per_region = 0
    zone_index = 0
    instances = []
    instance_names = []
    disk_names = []
    instance_results = []
    num_zones = len(config['compute']['zones'])

    for chunk in chunks:
        # Create instance-specific configuration
        dataset = chunk['dataset']
        chunk_start, chunk_end = chunk['chunk']

        curr_instance_name = instance_name % num_instances
        curr_disk_name = disk_name % num_instances

        # Create instance metadata
        metadata=[
            {'key': 'config', 'value': config.file_contents},
            {'key': 'instance_name', 'value': curr_instance_name},
            {'key': 'project_name', 'value': config['project']},
            {'key': 'bucket_name', 'value': bucket},
            # chunking metadata
            {'key': 'dataset', 'value': dataset},
            {'key': 'chunk_start', 'value': chunk_start},
            {'key': 'chunk_end', 'value': chunk_end},
            {'key': 'run_script', 'value': run_script}
            ]

        # Create a new instance
        logging.info('Creating GCE instance %s' % curr_instance_name)
        instances.append(GceInstance(curr_instance_name, curr_disk_name, image_name, config['compute']['zones'][zone_index],
                                     metadata, config))

        # update loop info
        num_instances += 1
        instances_per_region += 1
        instance_names.append(curr_instance_name)
        disk_names.append(curr_disk_name)
        instance_console = ('https://console.developers.google.com/'
                            'project/nth-clone-620/compute/instancesDetail/'
                            'zones/us-central1-a/instances/%s/console#end') % curr_instance_name

        # switch to new region if known to be above quota
        if instances_per_region >= config['compute']['instance_quota']:
            instances_per_region = 0
            zone_index += 1

        if zone_index >= num_zones:
            logging.warning('Cannot create more instances! Capping experiment at %d instances.' %(num_instances))
            break

    # clear global q
    global instance_launch_queue
    while not instance_launch_queue.empty():
        instance_launch_queue.get()

    # launch all instances using multiprocessing
    launch_start_time = time.time()
    if config['num_processes'] == 1:
        for instance in instances:
            instance.start()
    else:
        pool = mp.Pool(min(config['num_processes'], len(instances)))
        pool.map(launch_instance, instances)
    logging.info('Done launching instances')

    # put instance launch names into a queue
    instance_results = []
    while not instance_launch_queue.empty():
        curr_instance_name = instance_launch_queue.get() 
        instance_results.append('%s.tar.gz' % curr_instance_name)

    # set up service
    result_dl_start_time = time.time()
    service_not_ready = True
    while service_not_ready:
        try:
            service = discovery.build('storage', config['compute']['api_version'], http=auth_http)
            req = service.objects().list(bucket=bucket)
            service_not_ready = False
        except (ValueError, Exception) as e:
            logging.info('Connection failed. Retrying...')

    completed_instance_results = []
    while instance_results:
        # Wait before checking again
        time.sleep(sleep_time)

        logging.info('Checking for completion...')
        try:
            resp = req.execute()
        except (ValueError, Exception) as e:
            logging.info('Connection failed. Retrying...')
            continue

        try:
            items = resp['items']
        except KeyError as e:
            logging.error(e)
            logging.error(resp)
            continue

        logging.info('Waiting on %s', instance_results)
        for item in items:
            if item['name'] in instance_results:
                completed_instance_results.append(item['name'])
                instance_results.remove(item['name'])
                logging.info('Instance %s completed!' % item['name'])

    # Delete the instances.
    if config['num_processes'] == 1:
        for instance in instances :
            instance.stop()
    else:
        pool = mp.Pool(min(config['num_processes'], len(instances)))
        pool.map(stop_instance, instances)
    logging.info('Done stopping instances')

    # Print running instances
    all_running_instances = []
    for zone in config['compute']['zones']:
        zone_instances = gce_helper.list_instances(zone)
        lines = ['These are your running instances in zone %s:' %(zone)]
        for zone_instance in zone_instances:
            logging.info(zone_instance['name'])
            lines.append('    ' + zone_instance['name'])
        if not zone_instances:
            lines.append('    (none)')
        zone_instances_text = '\n'.join(lines)
        all_running_instances.append(zone_instances_text)
        logging.info(zone_instances_text)

    # Download the results
    store_dir, instance_result_dirs = gcs_helper.retrieve_results(config['bucket'], completed_instance_results, instance_root)

    # Send the user an email
    message = EMAIL_NOTIFICATION % dict(
        instance_id=instance_id,
        instance_names='\n'.join(map(lambda n: '    ' + n, instance_names)),
        experiment_config=config_file,
        script_commands=config['compute']['startup_script'],
        listinstances_output='\n\n'.join(all_running_instances)
    )

    send_notification_email(message=message, config=config,
                            subject="Your experiment has completed.")


    # Run the results script TODO: move above the email
    result_agg_start_time = time.time()
    results_script_call = 'python %s %s %s' %(config['results_script'], config_file, store_dir)
    os.system(results_script_call)

    # get runtime
    end_time = time.time()
    total_runtime = end_time - start_time
    launch_prep_time = launch_start_time - launch_prep_start_time
    launch_time = result_dl_start_time - launch_start_time
    dl_time = result_agg_start_time - result_dl_start_time
    agg_time = end_time - result_agg_start_time

    logging.info('Total runtime: %f' %(total_runtime))
    logging.info('Prep time: %f' %(launch_prep_time))
    logging.info('Launch time: %f' %(launch_time))
    logging.info('Run and download time: %f' %(dl_time))
    logging.info('Result aggregation time: %f' %(agg_time))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(parents=[argparser])
    parser.add_argument('config')
    parser.add_argument('-s', '--sleep', type=int, default=120) # seconds to sleep before rechecking
    args = parser.parse_args()
    launch_experiment(args, args.sleep)
