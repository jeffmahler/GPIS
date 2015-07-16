# Copyright 2012 Google Inc. All Rights Reserved.
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

"""Google Compute Engine helper class.

Use this class to:
- Start an instance
- List instances
- Delete an instance
"""

__author__ = 'kbrisbin@google.com (Kathryn Hurley)'

import logging
try:
  import simplejson as json
except:
  import json
import IPython
import os
import sys
import time
import traceback

from apiclient.discovery import build
from apiclient.errors import HttpError
from httplib2 import HttpLib2Error
from oauth2client.client import AccessTokenRefreshError

import boto
import gce
import StringIO
import tarfile

GOOGLE_STORAGE = 'gs'
LOCAL_FILE = 'file'

class Gcs(object):
  """
  Allows users to access Google Cloud Storange
  """

  def __init__(self, auth_http, config, project_id=None):
    """Initialize the Gcs object.

    Args:
      auth_http: an authorized instance of httplib2.Http.
      project_id: the API console project name.
    """

    self.config = config

    self.service = build(
        'storage', self.config['compute']['api_version'], http=auth_http)

    self.project_id = None
    if not project_id:
      self.project_id = self.config['project']
    else:
      self.project_id = project_id

    self.cache_dir = self.config['cache_dir'] # directory to cache downloads

  def retrieve(self, bucket_name, file_name, dir_name):
    """
    Downloads a file from the specified bucket under the GCE project.

    Args:
       bucket_name: String name for the bucket
       file_name: String name of file to download
       dir_name: String name of subdir of cahce dir to store

    Returns:
       True or false depending on success of download
    """
    if not bucket_name:
        raise ValueError('bucket_name required.')
    if not file_name:
        raise ValueError('file_name required.')

    # create output directory
    out_dir = os.path.join(config['cache_dir'], dir_name)
    if not os.path.exists(out_dir):
      os.mkdir(out_dir)

    # everything with system calls because the apis are overly complicated and confusing
    os.system('gsutil cp gs://%s/%s %s' %(bucket_name, file_name, out_dir))
    
    # check that file was downloaded
    local_file_name = os.path.join(self.cache_dir, file_name)
    if not os.path.exists(local_file_name):
        logging.error('Failed to download %s from bucket %s!' %(file_name, bucket_name))
        return False
    return True

  def retrieve_results(self, bucket_name, results_list, dir_name):
      """
      Retrieves and extracts an entire list of results (which must be .tar.gz)
      """
      result_dirs = []
      # download each result
      for result_name in results_list:
          logging.info('Retrieving %s' %(result_name))

          # retrive file
          if self.retrieve(bucket_name, result_name, dir_name):
              local_file_name = os.path.join(self.cache_dir, result_name)
              local_file_root, ext = os.path.splitext(local_file_name)

              # check extensions and extract
              if ext == '.gz':
                  local_file_root, ext = os.path.splitext(local_file_root)

                  if ext == '.tar':
                      tar_file = tarfile.open(local_file_name)
                      local_result_dir = os.path.join(self.cache_dir, local_file_root)
                      tar_file.extractall(local_result_dir)
                      result_dirs.append(local_result_dir)
      return result_dirs

  def _blocking_call(self, response, finished_status='DONE'):
    """Blocks until the operation status is done for the given operation.

    Args:
      response: The response from the API call.

    Returns:
      Dictionary response representing the operation.
    """

    status = response['status']

    while status != finished_status and response:
      operation_id = response['name']
      if 'zone' in response:
        zone = response['zone'].rsplit('/', 1)[-1]
        request = self.service.zoneOperations().get(
            project=self.project_id, zone=zone, operation=operation_id)
      else:
        request = self.service.globalOperations().get(
            project=self.project_id, operation=operation_id)
      response = self._execute_request(request)
      if response:
        status = response['status']
        logging.info(
          'Waiting until operation is %s. Current status: %s',
          finished_status, status)
        if status != finished_status:
          time.sleep(3)

    return response

  def _execute_request(self, request):
    """Helper method to execute API requests.

    Args:
      request: The API request to execute.

    Returns:
      Dictionary response representing the operation if successful.

    Raises:
      ApiError: Error occurred during API call.
    """

    try:
      response = request.execute()
    except AccessTokenRefreshError, e:
      logging.error('Access token is invalid.')
      raise gce.ApiError(e)
    except HttpError, e:
      logging.error('Http response was not 2xx.')
      raise gce.ApiError(e)
    except HttpLib2Error, e:
      logging.error('Transport error.')
      raise gce.ApiError(e)
    except Exception, e:
      logging.error('Unexpected error occured.')
      traceback.print_stack()
      raise gce.ApiError(e)

    return response

