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
import sys
import time

from apiclient import discovery
import httplib2
from oauth2client.client import flow_from_clientsecrets
from oauth2client.file import Storage
from oauth2client.tools import argparser, run_flow
from oauth2client import tools
import smtplib
import gce

import experiment_config as ec

INSTANCE_NAME_LENGTH = 10

INSERT_ERROR = 'Error inserting %(name)s.'
DELETE_ERROR = """
Error deleting %(name)s. %(name)s might still exist; You can use
the console (http://cloud.google.com/console) to delete %(name)s.
"""
EMAIL_NOTIFICATION = """
Your experiment %(instance_id)s has completed.

Here was the config used to run the experiment:

    %(experiment_config)s

Here is the set of scripts run:

    %(script_commands)s

Here's the output of `gcloud compute instances list`:

%(listinstances_output)s
"""

def delete_resource(delete_method, *args):
    """
    Delete a Compute Engine resource using the supplied method and args.

    Args:
      delete_method: The gce.Gce method for deleting the resource.
    """
    resource_name = args[0]
    logging.info('Deleting %s' % resource_name)

    try:
        delete_method(*args)
    except (gce.ApiError, gce.ApiOperationError, ValueError) as e:
        logging.error(DELETE_ERROR, {'name': resource_name})
        logging.error(e)

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
    instance_name = 'experiment-%s-' % instance_id + '%d'
    disk_name = instance_name + '-disk'
    image_name = config['compute']['image']

    # Initialize gce.Gce.
    logging.info('Initializing GCE')
    gce_helper = gce.Gce(auth_http, config, project_id=config['project'])

    # Start an instance for each chunk
    num_instances = 0
    instance_names = []
    disk_names = []
    instance_results = []
    for chunk in config['chunks']:
        dataset = chunk['dataset']
        chunk_start, chunk_end = chunk['chunk']

        curr_instance_name = instance_name % num_instances
        curr_disk_name = disk_name % num_instances

        # Start an instance with a local start-up script and boot disk.
        logging.info('Starting GCE instance %s' % curr_instance_name)
        try:
            gce_helper.start_instance(
                curr_instance_name, curr_disk_name, image_name,
                service_email=config['compute']['service_email'],
                scopes=config['compute']['scopes'],
                startup_script=config['compute']['startup_script'],
                metadata=[
                    {'key': 'config', 'value': config.file_contents},
                    {'key': 'instance_name', 'value': instance_name},
                    {'key': 'project_name', 'value': config['project']},
                    {'key': 'bucket_name', 'value': bucket},
                    # chunking metadata
                    {'key': 'dataset', 'value': dataset},
                    {'key': 'chunk_start', 'value': chunk_start},
                    {'key': 'chunk_end', 'value': chunk_end},
                ],
                additional_ro_disks=config['compute']['data_disks']
            )
        except (gce.ApiError, gce.ApiOperationError, ValueError, Exception) as e:
            # Delete the disk in case the instance fails to start.
            delete_resource(gce_helper.delete_disk, disk_name)
            logging.error(INSERT_ERROR, {'name': instance_name})
            logging.error(e)
            return
        except gce.DiskDoesNotExistError as e:
            logging.error(INSERT_ERROR, {'name': instance_name})
            logging.error(e)
            return

        num_instances += 1
        instance_names.append(curr_instance_name)
        disk_names.append(curr_disk_name)
        instance_results.append('%s.tar.gz' % curr_instance_name)
        instance_console = ('https://console.developers.google.com/'
                            'project/nth-clone-620/compute/instancesDetail/'
                            'zones/us-central1-a/instances/%s/console#end') % curr_instance_name
        logging.info('Instance is running! Check it out: %s' % instance_console)

    # set up service
    service_not_ready = True
    while service_not_ready:
        try:
            service = discovery.build('storage', config['compute']['api_version'], http=auth_http)
            req = service.objects().list(bucket=bucket)
            service_not_ready = False
        except (ValueError, Exception) as e:
            logging.info('Connection failed. Retrying...')

    while instance_results:
        # Wait before checking again
        time.sleep(sleep_time)

        logging.info('Checking for completion...')
        try:
            resp = req.execute()
        except (ValueError, Exception) as e:
            logging.info('Connection failed. Retrying...')
            continue

        items = resp['items']
        for item in items:
            if item['name'] in instance_results:
                instance_results.remove(item['name'])
                logging.info('Instance %s completed!' % item['name'])

    # Delete the instances.
    for curr_instance_name in instance_names:
        delete_resource(gce_helper.stop_instance, curr_instance_name)

    # Delete the disk.
    for curr_disk_name in disk_names:
        delete_resource(gce_helper.delete_disk, curr_disk_name)

    instances = gce_helper.list_instances()
    instance_list = ''
    for instance in instances:
        logging.info(instance['name'])
        instance_list += '\t' + instance['name'] + '\n'
    if not instances:
        instance_list = '\t(none)'
    logging.info('These are your running instances:')
    logging.info(instance_list)

    # Send the user an email
    message = EMAIL_NOTIFICATION % dict(
        instance_id=instance_id,
        experiment_config=config_file,
        script_commands=config['compute']['startup_script'],
        listinstances_output=instance_list
    )

    send_notification_email(message=message, config=config,
                            subject="Your experiment has completed.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(parents=[argparser])
    parser.add_argument('config')
    parser.add_argument('-s', '--sleep', type=int, default=120) # seconds to sleep before rechecking
    args = parser.parse_args()
    launch_experiment(args, args.sleep)
