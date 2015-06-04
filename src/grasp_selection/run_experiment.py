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
    message = "From: %s\nTo: %s\nSubject: %s\n\n%s" % (from_email, ", ".join(to_emails), subject, message)
    #server = smtplib.SMTP(SERVER)
    server = smtplib.SMTP("smtp.gmail.com", 587) #or port 465 doesn't seem to work!
    server.ehlo()
    server.starttls()
    server.login(gmail_user, gmail_pwd)
    server.sendmail(from_email, to_emails, message)
    #server.quit()
    server.close()
    print 'successfully sent the mail'

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
    instance_name = 'experiment-' + random_string(INSTANCE_NAME_LENGTH)
    disk_name = instance_name + '-disk'
    image_name = config['compute']['image']

    # Initialize gce.Gce.
    logging.info('Initializing GCE')
    gce_helper = gce.Gce(auth_http, config, project_id=config['project'])

    # Start an instance with a local start-up script and boot disk.
    logging.info('Starting GCE instance %s' %(instance_name))

    try:
      gce_helper.start_instance(
          instance_name, disk_name, image_name,
          service_email=config['compute']['service_email'],
          scopes=config['compute']['scopes'],
          startup_script=config['compute']['startup_script'],
          metadata=[
              {'key': 'config', 'value': config.file_contents},
              {'key': 'instance_name', 'value': instance_name},
              {'key': 'project_name', 'value': config['project']},
              {'key': 'bucket_name', 'value': bucket}
          ],
          additional_disks=config['compute']['data_disks']
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

    instance_console = 'https://console.developers.google.com/project/nth-clone-620/compute/' + \
                       'instancesDetail/zones/us-central1-a/instances/%s/console#end' % instance_name
    logging.info('Instance is running! Check it out: %s' % instance_console)

    instance_completed = False
    bucket_name = config['bucket']
    instance_data = '%s.tar.gz' %(instance_name)

    # set up service
    service_not_ready = True
    while service_not_ready:
      try:
        service = discovery.build('storage', config['compute']['api_version'], http=auth_http)
        req = service.objects().list(bucket=bucket_name)
        service_not_ready = False
      except (ValueError, Exception) as e:
        logging.info('Connection failed. Retrying...')

    while not instance_completed:
        # Wait before checking again
        time.sleep(sleep_time)

        logging.info('Checking for completion...')

        # List all running instances.
        try:
          resp = req.execute()
        except (ValueError, Exception) as e:
          logging.info('Connection failed. Retrying...')
          continue

        items = resp['items']
        for item in items:
          if item['name'] == instance_data:
            instance_completed = True

    # Delete the instance.
    delete_resource(gce_helper.stop_instance, instance_name)

    # Delete the disk.
    delete_resource(gce_helper.delete_disk, disk_name)

    logging.info('These are your running instances:')
    instances = gce_helper.list_instances()
    instance_list = ''
    for instance in instances:
        logging.info(instance['name'])
        instance_list += instance['name'] + '\n'

    # Send the user an email
    message = """
Your experiment %(experiment_name)s has completed.

Here was the config used to run the experiment:

%(experiment_config)s

Here is the set of scripts run:

%(script_commands)s

Here's the output of "gcutil listinstances":

%(listinstances_output)s
        """%dict(experiment_name=instance_name,
                 experiment_config=config_file,
                 script_commands=config['compute']['startup_script'],
                 listinstances_output = instance_list)

    send_notification_email(message=message, config=config,
                            subject="Your experiment has completed.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(parents=[argparser])
    parser.add_argument('config')
    parser.add_argument('-s', '--sleep', type=int, default=120) # seconds to sleep before rechecking
    args = parser.parse_args(sys.argv[1:])
    launch_experiment(args, args.sleep)
