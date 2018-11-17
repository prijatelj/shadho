"""Main driver for the SHADHO framework.

Classes
-------
Shadho
    Driver class for local and distributed hyperparameter optimization.
"""
from .config import ShadhoConfig
from .hardware import ComputeClass
from .managers import create_manager

import copy
from collections import OrderedDict
import json
import os
import tarfile
import tempfile
import time

import numpy as np
import pyrameter
import scipy.stats


def shadho():
    pass


class Shadho(object):
    """Optimize hyperparameters using specified hardware.

    Parameters
    ----------
    cmd : str or function
        The command to run on remote workers or function to run locally.
    spec : dict
        The specification defining search spaces.
    files : list of str or WQFile
        The files to send to remote workers for task execution.
    use_complexity : bool, optional
        If True, use the complexity heuristic to adjust search proportions.
    use_priority : bool, optional
        If True, use the priority heuristic to adjust search proportions.
    timeout : int, optional
        Number of seconds to search for.
    max_tasks : int, optional
        Number of tasks to queue at a time.
    await_pending : bool, optional
        If True, wait for all running tasks to complete after `timeout`.
        Default: False
    max_resubmissions: int, optional
        Maximum number of times to resubmit a particular parameterization for
        processing if task failure occurs. Default is not to resubmit.

    Attributes
    ----------
    config : `shadho.config.ShadhoConfig`
        Global configurations for shadho.
    backend : `pyrameter.ModelGroup`
        The data view/storage backend. This backend keeps track of all
    manager : `shadho.managers.workqueue.WQManager`
        The distributed task manager to use.
    ccs : list of `shadho.hardware.ComputeClass`
        The types of hardware to expect during optimization. If not supplied,
        tasks are run on the first available worker.
    use_complexity : bool
        If True, use the complexity heuristic to adjust search proportions.
    use_priority : bool
        If True, use the priority heuristic to adjust search proportions.
    timeout : int
        Number of seconds to search for.
    max_tasks : int
        Maximum number of tasks to enqueue at a time.
    await_pending : bool
        If True, wait for all running tasks to complete after `timeout`.
    max_resubmissions: int
        Maximum number of times to resubmit a particular parameterization for
        processing if task failure occurs. Default is not to resubmit.

    Notes
    -----
    To enable configuration after intialization, ``backend`` and ``manager``
    are created when `Shadho.run` is called.

    """

    def __init__(self, cmd, spec, backend=None, files=None, use_complexity=True,
                 use_priority=True, timeout=600, max_tasks=100,
                 await_pending=False, max_resubmissions=0):
        self.config = ShadhoConfig()
        self.cmd = cmd
        self.spec = spec
        self.use_complexity = use_complexity
        self.use_priority = use_priority
        self.timeout = timeout if timeout is not None and timeout >= 0 \
                       else float('inf')
        self.max_tasks = 2 * max_tasks
        self.max_resubmissions = max_resubmissions
        self.await_pending = await_pending

        self.ccs = OrderedDict()

        self.files = []
        if files is not None:
            for f in files:
                if not isinstance(f, WQFile):
                    self.add_input_file(f)
                else:
                    self.files.append(f)
        self.assignments = {}

        self.__tmpdir = tempfile.mkdtemp(prefix='shadho_', suffix='_output')

        self.add_input_file(os.path.join(
            self.config.shadho_dir,
            self.config.wrapper))

        self.config.save_config(self.__tmpdir)
        self.add_input_file(os.path.join(self.__tmpdir, '.shadhorc'))
        self.backend = backend

    def __del__(self):
        if hasattr(self, '__tmpdir') and self.__tmpdir is not None:
            os.rmdir(self.__tmpdir)

    def add_input_file(self, localpath, remotepath=None, cache=True):
        """Add an input file to the global file list.

        Parameters
        ----------
        localpath : str
            Path to the file on the local filesystem.
        remotepath : str, optional
            Path to write the file to on the remote worker. If omitted, the
            basename of ``localpath`` (e.g. "foo/bar.baz" => "bar.baz").
        cache : bool, optional
            Whether to cache the file on the remote worker. If True (default),
            will be cached on the worker between tasks, reducing network
            transfer overhead. If False, will be re-transferred to the worker
            on each task.
        """
        self.files.append((localpath, remotepath, 'input', cache))

    def add_output_file(self, localpath, remotepath=None, cache=False):
        """Add an input file to the global file list.

        Output files are expected to be discovered on the remote worker after a
        task has completed. They are returned to the `shadho.Shadho` instance
        and will be stored for further review without additional processing.

        Parameters
        ----------
        localpath : str
            Path to the file on the local filesystem.
        remotepath : str, optional
            Path to write the file to on the remote worker. If omitted, the
            basename of ``localpath`` (e.g. "foo/bar.baz" => "bar.baz").
        cache : bool, optional
            Whether to cache the file on the remote worker. It is recommended
            that this be set to False for output files.

        Notes
        -----
        `shadho.Shadho` automatically parses the output file specified in
        ``.shadhorc``, so and output file added through this method will not be
        processed, but rather stored for later review.
        """
        self.files.append((localpath, remotepath, 'output', cache))

    def add_compute_class(self, name, resource, value, max_tasks=100):
        """Add a compute class representing a set of consistent recources.

        Parameters
        ----------
        name : str
            The name of this set of compute resources.
        resource : str
            The resource to match, e.g. gpu_name, cores, etc.
        value
            The value of the resource that should be matched, e.g. "TITAN X
            (Pascal)", 8, etc.
        max_tasks : int, optional
            The maximum number of tasks to queue for this compute class,
            default 100.
        """
        cc = ComputeClass(name, resource, value, 2 * max_tasks)
        self.ccs[cc.id] = cc

    def run(self):
        """Search hyperparameter values on remote workers.

        Generate and evaluate hyperparameters using the selected task manager
        and search strategy. Hyperparameters will be evaluated until timeout,
        and the optimal set will be printed to screen.

        Notes
        -----
        If `self.await_pending` is True, Shadho will continue to evaluate
        hyperparameters in the queue without generating new hyperparameter
        values. This will continue until the queue is empty and all tasks have
        returned.
        """
        # Set up the task manager as defined in `shadho.managers`
        if not hasattr(self, 'manager'):
            self.manager = create_manager(
                manager_type=self.config.manager,
                config=self.config,
                tmpdir=self.__tmpdir)

        # Set up the backend hyperparameter generation and database
        if not hasattr(self, 'backend') or self.backend is None:
            self.backend = pyrameter.build(self.spec,
                                           db=self.backend,
                                           complexity_sort=self.use_complexity,
                                           priority_sort=self.use_priority)

        # If no ComputeClass was created, create a dummy class.
        if len(self.ccs) == 0:
            cc = ComputeClass('all', None, None, self.max_tasks)
            self.ccs[cc.id] = cc

        # Set up intial model/compute class assignments.
        self.assign_to_ccs()

        start = time.time()
        elapsed = 0
        try:
            # Run the search until timeout or until all tasks complete
            while elapsed < self.timeout and (elapsed == 0 or not self.manager.empty()):
                # Generate hyperparameters and a flag to continue or stop
                stop = self.generate()
                if not stop:
                    # Run another task and await results
                    result = self.manager.run_task()
                    if result is not None:
                        # If a task returned post-process as a success or fail
                        if len(result) == 3:
                            self.success(*result)  # Store and move on
                        else:
                            self.failure(*result)  # Resubmit if asked
                    # Checkpoint the results to file or DB at some frequency
                    if self.backend.result_count % 50 == 0:
                        self.backend.save()
                    # Update the time for timeout check
                    elapsed = time.time() - start
                else:
                    break

            # If requested, continue the loop until all tasks return
            if self.await_pending:
                while not self.manager.empty():
                    result = self.manager.run_task()
                    if result is not None:
                        if len(result) == 4:
                            self.success(*result)
                        else:
                            self.failure(*result)

        # On keyboard interrupt, save any results and clean up
        except KeyboardInterrupt:
            if hasattr(self, '__tmpdir') and self.__tmpdir is not None:
                os.rmdir(self.__tmpdir)

        # Save the results and print the optimal set of parameters to  screen
        self.backend.save()
        opt = self.backend.optimal(mode='best')
        key = list(opt.keys())[0]
        print("Optimal result: {}".format(opt[key]['loss']))
        print("With parameters: {}".format(opt[key]['values']))
        print("And additional results: {}".format(opt[key]['results']))

    def generate(self):
        """Generate hyperparameter values to test.

        Hyperparameter values are generated from the search space specification
        supplied at instantiation using the requested generation method (i.e.,
        random search, TPE, Gaussian process Bayesian optimization, etc.).

        Returns
        -------
        stop : bool
            If True, no values were generated and the search should stop. This
            facilitates grid-search-like behavior, for example stopping on
            completion of an exhaustive search.

        Notes
        -----
        This method will automatically add a new task to the queue after
        generating hyperparameter values.
        """
        stop = True

        # Generate hyperparameters for every compute class with space in queue
        for cc_id in self.ccs:
            cc = self.ccs[cc_id]
            n = cc.max_tasks - cc.current_tasks

            # Generate enough hyperparameters to fill the queue
            for i in range(n):
                # Get bookkeeping ids and hyperparameter values
                model_id, result_id, param = cc.generate()

                # Create a new distributed task if values were generated
                if param is not None:
                    # Encode info to map to db in the task tag
                    tag = '.'.join([result_id, model_id, cc_id])
                    param_copy = copy.deepcopy(param)
                    for kernel in param_copy:
                        param_copy[kernel]['cores'] = cc.value
                    self.manager.add_task(
                        self.cmd,
                        tag,
                        param_copy,
                        files=self.files,
                        resource=cc.resource,
                        value=cc.value)
                    stop = False  # Ensure that the search continues
            cc.current_tasks = cc.max_tasks  # Update to show full queue

    def assign_to_ccs(self):
        """Assign trees to compute classes.

        Each independent model in the search (model being one of a disjoint set
        of search domains) is assigned to at least two compute classes based on
        its rank relative to other models. In this way, only a subset of models
        are evaluated on each set of hardware.

        Notes
        -----
        This method accounts for differing counts of models and compute
        classes, adjusting for a greater number of models, a greater number of
        compute classes, or equal counts of models and compute classes.

        See Also
        --------
        `shadho.ComputeClass`
        `pyrameter.ModelGroup`
        """
        # NEWFANGLED WAY: assign COPIES of ALL models to ALL compute classes
        for key in list(self.ccs.keys()):
            self.ccs[key].clear()
            for mid in self.backend.model_ids:
                self.ccs[key].add_model(self.backend[mid].copy(parent_inherits_results=True))

        """
        # If only one CC exists, do nothing; otherwise, update assignments
        if len(self.ccs) == 1:
            key = list(self.ccs.keys())[0]
            self.ccs[key].model_group = self.backend
        else:
            # Sort models in the search by complexity, priority, or both and
            # get the updated order.
            self.backend.sort_models()
            model_ids = [mid for mid in self.backend.model_ids]

            # Clear the current assignments
            for key in list(self.ccs.keys()):
                self.ccs[key].clear()

            # Determine if the number of compute classes or the number of
            # model ids is larger
            ccids = list(self.ccs.keys())
            larger = model_ids if len(model_ids) >= len(ccids) else ccids
            smaller = ccids if larger == model_ids else model_ids

            # Assign models to CCs such that each model is assigned to at
            # least two CCs.

            # Steps between `smaller` index increment
            x = float(len(larger)) / float(len(smaller))
            y = x - 1  # Current step index (offset by 1 for 0-indexing)
            j = 0  # Current index of `smaller`
            m = len(smaller) / 2  # Halfway point for second assignment
            n = len(larger) / 2  # Halfway point for second assignment

            for i in range(len(larger)):
                # If at a step point for `smaller` increment the index
                if i > np.ceil(y):
                    j += 1
                    y += x

                # Add the model to the current CC.
                # If cc_idx < half_of_num_ccs:
                #     add model to cc at cc_idx + 1.
                # Else:
                #     add model to cc at cc_idx - 1
                if smaller[j] in self.ccs:
                    self.ccs[smaller[j]].add_model(self.backend[larger[i]])
                    if j < m:
                        self.ccs[smaller[j + 1]].add_model(
                            self.backend[larger[i]])
                    else:
                        self.ccs[smaller[j - 1]].add_model(
                            self.backend[larger[i]])
                else:
                    self.ccs[larger[i]].add_model(self.backend[smaller[j]])
                    if i < n:
                        self.ccs[larger[i + 1]].add_model(
                            self.backend[smaller[j]])
                    else:
                        self.ccs[larger[i - 1]].add_model(
                            self.backend[smaller[j]])
        """

    def success(self, tag, loss, results):
        """Handle successful task completion.

        Parameters
        ----------
        tag : str
            The task tag, encoding the result id, model id, and compute class
            id as ``<result_id>.<model_id>.<cc_id>``.
        loss : float
            The loss value associated with this result.
        results : dict
            Additional metrics to be included with this result.

        Notes
        -----
        This method will trigger a model/compute class reassignment in the
        event that storing the result caused the model's priority to be
        updated.
        """
        # Get bookkeeping information from the task tag
        result_id, model_id, ccid = tag.split('.')

        # Update the DB with the result
        #self.backend.register_result(model_id, result_id, loss, results)
        self.ccs[ccid].register_result(model_id, result_id, loss, results)

        # Reassign models to CCs at some frequency
        # if self.backend.result_count % 10 == 0:
        #     self.assign_to_ccs()

        # Update the number of enqueued items
        self.ccs[ccid].current_tasks -= 1

    def failure(self, tag, resub):
        """Handle task failure.

        Parameters
        ----------
        task : `work_queue.Task`
            The failed task to process.

        Notes
        -----
        This method will resubmit failed tasks on request to account for
        potential worker dropout, etc.
        """
        # Get bookkeeping information from the task tag
        result_id, model_id, ccid = tag.split('.')

        # Determine whether or not to resubmit
        submissions, params = \
            self.backend.register_result(model_id, result_id, None, {})

        # Resubmit the task if it should be, otherwise update the number of
        # enqueued items.
        if resub and submissions < self.max_resubmissions:
            cc = self.ccs[ccid]
            self.manager.add_task(self.cmd,
                                  tag,
                                  params,
                                  files=self.files,
                                  resource=cc.resource,
                                  value=cc.value)
        else:
            self.ccs[ccid].current_tasks -= 1
