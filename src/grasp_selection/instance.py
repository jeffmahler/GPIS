from abc import ABCMeta, abstractmethod

import gce
import IPython
import itertools as it
import logging
import multiprocessing as mp
import time

import httplib2
from oauth2client.client import flow_from_clientsecrets
from oauth2client.file import Storage
from oauth2client.tools import argparser, run_flow
from oauth2client import tools

import copy_reg
import types

def oauth_authorization(config):
    """
    Perform OAuth2 authorization and return an authorized instance of
    httplib2.Http.
    """
    # perform OAuth 2.0 authorization flow.
    flow = flow_from_clientsecrets(
        config['client_secrets'], scope=config['compute']['scopes'])
    storage = Storage(config['oauth_storage'])
    credentials = storage.get()

    # authorize an instance of httplib2.Http.
    logging.info('Authorizing Google API')
    if credentials is None or credentials.invalid:
        credentials = run_flow(flow, storage, args)
    http = httplib2.Http()
    auth_http = credentials.authorize(http)
    return auth_http

class VMInstanceAllocator(object):
    __metaclass__ = ABCMeta

    @staticmethod
    def partition_data(config):
        """
        Partition a dataset into chunks
        """
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
                             chunk_start=assigned,
                             chunk_end=assigned+max_chunk_size)
                chunks.append(chunk)
                assigned += max_chunk_size
        return chunks

    @staticmethod
    def partition_parameters_list(config):
        """
        Partition up the paarameter space by crossing a hardcoded list (not automatic to prevent overallocation, since things build up quickly)
        """
        params = []
        param_dict = config['param_values']

        # take the product of the paraeter values in a list
        for param_combination in it.product(*param_dict.values()):
            p = {}
            for param_name, param_val in zip(param_dict.keys(), param_combination):
                p[param_name] = param_val
            params.append(p)
        return params

    @staticmethod
    def partition(config, param_fns = None):
        """
        Partition the parameters into a list
        """
        # create list of functions to generate parameters
        if param_fns is None:
            param_fns = [VMInstanceAllocator.partition_data, VMInstanceAllocator.partition_parameters_list]
        out_params = []
        params = [fn(config) for fn in param_fns]

        # cross product of the parameters
        for param_tup in it.product(*params):
            all_param = {}
            for param in param_tup:
                all_param.update(param)
            out_params.append(all_param)
        return out_params        

    @abstractmethod
    def allocate():
        """
        Returns a list of instance objects with parameters from config.
        The config must be an object that can be accessed using string dictionary keys and containing the chunking / splitting params 
        """
        pass

class GceInstanceAllocator(VMInstanceAllocator):
    def __init__(self, config, gce_helper):
        self.config = config
        self.gce_helper = gce_helper
        
    def instance_limits(self, use_hard_limits=False):
        """
        Return a dictionary of the number of available instances to allocate per zone
        """
        instance_quota = self.config['compute']['instance_quota']
        available_instances = {}
        for zone in self.config['compute']['zones']:
            zone_instances = self.gce_helper.list_instances(zone)
            available_instances[zone] = instance_quota
            if use_hard_limits:
                available_instances[zone] = instance_quota - len(zone_instances)
        return available_instances

    def allocate(self, instance_root, run_script, data_disks, use_hard_limits=True):
        """
        Returns a list of GCE instance objects with parameters from config.
        The config must be an object that can be accessed using string dictionary keys and containing the chunking / splitting params 
        """
        # generate a parameter list
        param_list = VMInstanceAllocator.partition(self.config)

        # get available resources
        instance_lims = self.instance_limits(use_hard_limits)

        # init GCE config stuff
        compute_config = self.config['compute']
        instances_per_zone = 0
        instances = []
        instance_name = '%s-' %(instance_root) + '%d'
        disk_name = '%s-' %(instance_root) + '%d-disk'

        zone_index = 0
        zones = compute_config['zones']
        project = compute_config['project']
        bucket = compute_config['bucket']
        image = compute_config['image']
        cur_zone = zones[zone_index]

        # loop through the params
        for instance_num, params in enumerate(param_list):
            # create instance-specific configuration
            cur_instance_name = instance_name % instance_num
            cur_disk_name = disk_name % instance_num

            # create instance metadata
            metadata=[
                {'key': 'config', 'value': self.config.file_contents},
                {'key': 'job_root', 'value': self.config['job_root']},
                {'key': 'instance_name', 'value': cur_instance_name},
                {'key': 'run_script', 'value': run_script},
                {'key': 'project_name', 'value': compute_config['project']},
                {'key': 'bucket_name', 'value': compute_config['bucket']}
                ]
            for param_key, param_val in params.items():
                metadata.append({'key': param_key, 'value': param_val})

            # create a new instance
            logging.info('Allocating GCE instance %s in zone %s' %(cur_instance_name, zones[zone_index]))
            instances.append(GceInstance(cur_instance_name, cur_disk_name, image, zones[zone_index],
                                         metadata, [data_disks[zone_index]], project, self.config))
            instances_per_zone += 1

            # switch to new region if known to be above quota
            if instances_per_zone >= instance_lims[cur_zone]:
                instances_per_zone = 0
                zone_index += 1

                # check for zone overflows
                if zone_index >= len(zones):
                    logging.warning('Cannot create more instances! Capping allocation at %d instances.' %(instance_num+1))
                    return instances

                # update current zone
                cur_zone = zones[zone_index]

        # return list of instances
        return instances

# helper functions for multiprocessing
def _pickle_method(m):
    if m.im_self is None:
        return getattr, (m.im_class, m.im_func.func_name)
    else:
        return getattr, (m.im_self, m.im_func.func_name)

copy_reg.pickle(types.MethodType, _pickle_method)
MAX_QUEUE_SIZE = 1000

class VMInstanceManager(object):
    instance_launch_queue = mp.Queue(MAX_QUEUE_SIZE)

    def __init__(self, queue_timeout=30.):
        self.queue_timeout = queue_timeout

    def launch_instances(self, instances, num_processes=1):
        """ Launch a list of instances """
        # clear instance queue
        while not VMInstanceManager.instance_launch_queue.empty():
            VMInstanceManager.instance_launch_queue.get()

        # launch everything
        if num_processes == 1:
            for instance in instances:
                self._launch_instance(instance)
        else:
            pool = mp.Pool(min(num_processes, len(instances)))
            pool.map(self._launch_instance, instances)

        # return dictionary of launched instances mapped by name (some may fail)
        time.sleep(1) # wait for queue population, which sometimes fucks up
        launched_instances = {}
        while not VMInstanceManager.instance_launch_queue.empty():
            cur_instance = VMInstanceManager.instance_launch_queue.get() 
            launched_instances[cur_instance.instance_name] = cur_instance
        return launched_instances

    def stop_instances(self, instances, num_processes=1):
        """ Stop a list of instances """
        if len(instances) == 0:
            return
        
        if num_processes == 1:
            for instance in instances:
                instance.stop()
        else:
            pool = mp.Pool(min(num_processes, len(instances)))
            pool.map(self._stop_instance, instances)

    def _launch_instance(self, instance):
        """ Launches an instance object. Static method for use in multiprocessing. """
        if not isinstance(instance, VMInstance):
            raise ValueError('Must provide an instance object to start')
        try:
            if instance.start():
                VMInstanceManager.instance_launch_queue.put(instance, timeout=self.queue_timeout)
        except:
            logging.info('Failed to launch %s' %(instance.instance_name))

    def _stop_instance(self, instance):
        """ Stops an instance object. """
        if not isinstance(instance, VMInstance):
            raise ValueError('Must provide an instance object to stop')
        instance.stop()
                
class VMInstance(object):
    """ Abstract class to wrap per-instance configurations and starting / stopping of virtual instances """
    __metaclass__ = ABCMeta

    @abstractmethod
    def start(self):
        """ Starts a virtual instance """
        pass

    @abstractmethod
    def stop(self):
        """ Shuts down a virtual instance """
        pass

class GceInstance(VMInstance):
    """ An instance object for GCE """
    def __init__(self, instance_name, disk_name, image_name, zone, metadata, data_disks, project, config):
        self.instance_name = instance_name
        self.disk_name = disk_name
        self.image_name = image_name
        self.zone = zone
        self.metadata = metadata
        self.data_disks = data_disks
        self.project = project
        self.config = config
        self.running = False

    def create_gce_helper(self):
        """ Creates a local gce helper to re-authorize with google """
        auth_http = oauth_authorization(self.config)
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
                additional_disks = self.data_disks
            )
        except (gce.ApiError, gce.ApiOperationError, ValueError, Exception) as e:
            # Delete the disk in case the instance fails to start.
            logging.error(INSERT_ERROR, {'name': self.instance_name})
            logging.error(e)
            self.delete_disk()
            return False
        except gce.DiskDoesNotExistError as e:
            logging.error(INSERT_ERROR, {'name': self.instance_name})
            logging.error(e)
            return False

        instance_console = ('https://console.developers.google.com/'
                            'project/%s/compute/instancesDetail/'
                            'zones/%s/instances/%s/console#end') % (self.project, self.zone, self.instance_name)
        logging.info('Instance %s is running! Check it out: %s' %(self.instance_name, instance_console))        
        self.running = True
        return True

    def stop(self):
        """ Stop and cleanup this instance """
        # stop instance
        self.terminate_instance()
        # delete disk
        self.delete_disk()
        self.running = False

