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
from scipy.optimize import linprog
import pprint


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
                 await_pending=False, max_resubmissions=0, sort_method=None):
        self.config = ShadhoConfig()
        self.cmd = cmd
        self.spec = spec
        self.use_complexity = use_complexity
        self.use_priority = use_priority
        self.sort_method = sort_method
        self.timeout = timeout if timeout is not None and timeout >= 0 \
                       else float('inf')
        self.max_tasks = 2 * max_tasks
        self.max_resubmissions = max_resubmissions
        self.await_pending = await_pending

        self.ccs = OrderedDict()

        self.fake_ccs = OrderedDict()
        self.fake_to_real_ccids = {}
        self.real_to_fake_ccids = {}
        self.first_assignment = True
        self.sched_data = {}
        self.pp = pprint.PrettyPrinter(indent=2)
        self.first_modify = True

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

        # self.global_work_percent_targets = None

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
        cc2 = ComputeClass(name, resource, value, 2 * max_tasks)
        self.ccs[cc.id] = cc
        self.fake_ccs[cc2.id] = cc2
        self.fake_to_real_ccids[cc2.id] = cc.id
        self.real_to_fake_ccids[cc.id] = cc2.id

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
                                           priority_sort=self.use_priority)#,
                                           #sort_method=self.sort_method)

        # If no ComputeClass was created, create a dummy class.
        if len(self.ccs) == 0:
            cc = ComputeClass('all', None, None, self.max_tasks)
            self.ccs[cc.id] = cc
            self.fake_ccs[cc.id] = cc

        for ccid in list(self.ccs.keys()):
            self.sched_data[ccid] = {}
            for mid in self.backend.model_ids:
                self.sched_data[ccid][mid] = {'avg_runtime': 100000000,
                                              'tot_runtime': 0,
                                              'num_runs': 0,
                                              'speedup': 1.0,
                                              'per_compute_class_rank': 0,
                                              'jhibshma_rank': 1}

        # Set up intial model/compute class assignments.
        self.assign_to_ccs()
        mids = list(self.backend.model_ids)
        # target_probs = [0.40, 0.18, 0.20, 0.22]
        # self.global_work_percent_targets = {}
        # for mid_idx in range(len(mids)):
        #     self.global_work_percent_targets[mids[mid_idx]] = target_probs[mid_idx]
        self.modify_probabilities(fake_cc_use='Copy')

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
                        # Don't forget the - 1
                        param_copy[kernel]['cores'] = cc.value - 1
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
        if self.first_assignment:
            self.first_assignment = False
            for key in list(self.ccs.keys()):
                self.ccs[key].clear()
                for mid in self.backend.model_ids:
                    self.ccs[key].add_model(self.backend[mid].copy(parent_inherits_results=True))

        # OLD WAY: only do this with the fake_ccs for prob calculations.
        # If only one CC exists, do nothing; otherwise, update assignments
        if len(self.fake_ccs) == 1:
            key = list(self.fake_ccs.keys())[0]
            self.fake_ccs[key].model_group = self.backend
        else:
            # Sort models in the search by complexity, priority, or both and
            # get the updated order.
            self.backend.sort_models()
            model_ids = [mid for mid in self.backend.model_ids]

            # Clear the current assignments
            for key in list(self.fake_ccs.keys()):
                self.fake_ccs[key].clear()

            # Determine if the number of compute classes or the number of
            # model ids is larger
            ccids = list(self.fake_ccs.keys())
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
                if smaller[j] in self.fake_ccs:
                    self.fake_ccs[smaller[j]].add_model(self.backend[larger[i]])
                    if j < m:
                        self.fake_ccs[smaller[j + 1]].add_model(
                            self.backend[larger[i]])
                    else:
                        self.fake_ccs[smaller[j - 1]].add_model(
                            self.backend[larger[i]])
                else:
                    self.fake_ccs[larger[i]].add_model(self.backend[smaller[j]])
                    if i < n:
                        self.fake_ccs[larger[i + 1]].add_model(
                            self.backend[smaller[j]])
                    else:
                        self.fake_ccs[larger[i - 1]].add_model(
                            self.backend[smaller[j]])

    def update_sched_data(self, ccid, mid, results):
        """Update self.sched_data with a successful run's metrics.

        This includes a full re-computation of all the sched_data.
        """
        self.sched_data[ccid][mid]['num_runs'] += 1.0
        self.sched_data[ccid][mid]['tot_runtime'] += results['finish_time'] - results['start_time']
        self.sched_data[ccid][mid]['avg_runtime'] = self.sched_data[ccid][mid]['tot_runtime'] /\
                                                    float(self.sched_data[ccid][mid]['num_runs'])
        #print('A grand experiment:')
        #self.sched_data = {
        #    'a': {1: {'speedup': 1.0}, 2: {'speedup': 1.0}, 3: {'speedup': 1.0}, 4: {'speedup': 1.0}},
        #    'b': {1: {'speedup': 1.5}, 2: {'speedup': 1.4}, 3: {'speedup': 1.5}, 4: {'speedup': 1.3}},
        #    'c': {1: {'speedup': 2.4}, 2: {'speedup': 2.3}, 3: {'speedup': 2.3}, 4: {'speedup': 2.8}},
        #    'd': {1: {'speedup': 5.0}, 2: {'speedup': 5.1}, 3: {'speedup': 5.2}, 4: {'speedup': 4.9}}
        #}
        ccids = list(self.ccs.keys())
        ccids.sort()
        mids = list(self.backend.model_ids)
        mids.sort()

        # First update speedup for mid
        max_avg_runtime = 0
        for a_ccid in ccids:
            a = self.sched_data[a_ccid][mid]['avg_runtime']
            if a > max_avg_runtime:
                max_avg_runtime = a
        for a_ccid in ccids:
            self.sched_data[a_ccid][mid]['speedup'] = max_avg_runtime / self.sched_data[a_ccid][mid]['avg_runtime']

        # Second, update all per compute class ranks:
        #
        # The following code is a bit confusing.
        # Say we have speedups of [1.2, 2.1, 2.1, 1.7]
        # Then the ranks of these speedups are said to be [4, 1.5, 1.5, 3]
        # It is these ranks that the code calculates.
        # Here a higher speedup leads to a "lower" rank value.
        for a_ccid in ccids:
            speedups = []
            for a_mid in mids:
                speedups.append(self.sched_data[a_ccid][a_mid]['speedup'])
            speedups.sort(reverse=True)
            per_compute_class_ranks = []
            idx = 0
            for i in range(0, len(speedups) + 1):
                if i == len(speedups) or speedups[i] != speedups[idx]:
                    num_tied = i - idx
                    for j in range(idx, i):
                        per_compute_class_ranks.append((num_tied + 1.0) / 2.0 + idx)
                    idx = i
            for a_mid in mids:
                for i in range(0, len(speedups)):
                    if self.sched_data[a_ccid][a_mid]['speedup'] == speedups[i]:
                        self.sched_data[a_ccid][a_mid]['per_compute_class_rank'] = per_compute_class_ranks[i]
                        break

        # Third, update all jhibshma ranks:
        # Same idea as above except the transpose of it.
        # Here, "lower" per_compute_class_rank value corresponds to a lower jhibshma_rank value.
        for a_mid in mids:
            per_compute_class_ranks = []
            for a_ccid in ccids:
                per_compute_class_ranks.append(self.sched_data[a_ccid][a_mid]['per_compute_class_rank'])
            per_compute_class_ranks.sort()
            jhibshma_ranks = []
            idx = 0
            for i in range(0, len(per_compute_class_ranks) + 1):
                if i == len(per_compute_class_ranks) or per_compute_class_ranks[i] != per_compute_class_ranks[idx]:
                    num_tied = i - idx
                    for j in range(idx, i):
                        jhibshma_ranks.append((num_tied + 1.0) / 2.0 + idx)
                    idx = i
            for a_ccid in ccids:
                for i in range(0, len(per_compute_class_ranks)):
                    if self.sched_data[a_ccid][a_mid]['per_compute_class_rank'] == per_compute_class_ranks[i]:
                        self.sched_data[a_ccid][a_mid]['jhibshma_rank'] = jhibshma_ranks[i]
                        break

        #for a_ccid in ccids:
        #    for a_mid in mids:
        #        if self.sched_data[a_ccid][a_mid]['num_runs'] > 0:
        #            self.ccs[a_ccid].model_group.models[a_mid].jhibshma_rank =\
        #                self.sched_data[a_ccid][a_mid]['jhibshma_rank']
        #        else:  # If it's never run, give it super high priority:
        #            self.ccs[a_ccid].model_group.models[a_mid].jhibshma_rank = 0.01

        # Update schedule probs to match target probs.

    # global_dist_target: An array of percentages in order of sorted mid
    #   specifies what percent of completed jobs should be of the corresponding model
    # fake_cc_use:
    #   'None' --> Ignore fake ccs entirely
    #   'Copy' --> Act as if we're using the fake ccs
    #   'Modify' --> Get probabilities from fake ccs and then
    def modify_probabilities(self, global_dist_target=None, fake_cc_use='None'):
        if global_dist_target is not None:
            if fake_cc_use != 'None':
                print('Conflicting arguments passed: global_dist_target is not None and fake_cc_use is also not \'None\'.')
                print('Setting fake_cc_use to \'None\'')
            fake_cc_use = 'None'
        if fake_cc_use != 'None' and fake_cc_use != 'Copy' and fake_cc_use != 'Modify':
            print('Error: fake_cc_use has invalid value of \'' + fake_cc_use + '\'. Setting to \'None\'.')
            fake_cc_use = 'None'
        ccids = list(self.ccs.keys())
        ccids.sort()
        mids = list(self.backend.model_ids)
        mids.sort()
        num_ccs = len(ccids)
        num_models = len(mids)

        # Before even starting, check that a job has finished on each compute class.
        # If not, do no updates.
        have_not_run = 0
        max_avg_runtime = None
        min_avg_runtime = None
        have_not_run_on_all_ccs = {}
        for a_ccid in ccids:
            for a_mid in mids:
                if self.sched_data[a_ccid][a_mid]['num_runs'] == 0:
                    have_not_run += 1
                    have_not_run_on_all_ccs[a_mid] = 1
                    print('Missing Model ' + a_mid + ' on cc ' + a_ccid)
                else:
                    if max_avg_runtime is None or max_avg_runtime < self.sched_data[a_ccid][a_mid]['avg_runtime']:
                        max_avg_runtime = self.sched_data[a_ccid][a_mid]['avg_runtime']
                    if min_avg_runtime is None or min_avg_runtime > self.sched_data[a_ccid][a_mid]['avg_runtime']:
                        min_avg_runtime = self.sched_data[a_ccid][a_mid]['avg_runtime']
        if have_not_run > 0:
            # Used to have code which tells all things which _have_ run not to.
            print('')
            # return

        # If not everything has run yet and we're not just copying fake_ccs:
        if (fake_cc_use == 'None' or fake_cc_use == 'Modify') and self.first_modify:
            self.first_modify = False
            if have_not_run > 0:  # This check is redundant if using first_modify.
                for a_ccid in ccids:
                    for a_mid in mids:
                        if self.sched_data[a_ccid][a_mid]['num_runs'] == 0:
                            self.ccs[a_ccid].model_group.models[a_mid].modified_prob = 1
                        else:
                            self.ccs[a_ccid].model_group.models[a_mid].modified_prob = 0.001
                print('\"Manually\" setting probabilities.')
                return

        # Step 1: Get info nicely into matrices (row = cc, col = model)
        global_prob_matrix = []
        global_avg_matrix = []
        global_job_per_time_matrix = []
        global_percent_running_matrix = []

        for a_ccid in ccids:
            if fake_cc_use != 'None':
                compute_class_model_probabilities = self.fake_ccs[self.real_to_fake_ccids[a_ccid]].get_probabilities(modified=False)
                for a_mid in mids:
                    if a_mid not in compute_class_model_probabilities:
                        compute_class_model_probabilities[a_mid] = 0
            else:
                compute_class_model_probabilities = self.ccs[a_ccid].get_probabilities(modified=False)

            # The following modification IS NOT ACTUALLY CORRECT. NOR IS IT DOING ANYTHING.
            # Rather, it exists to give an estimate of the speedup we'll get.
            if global_dist_target is not None:
                compute_class_model_probabilities = global_dist_target
            prob_row = []
            avg_row = []
            job_per_time_row = []
            percent_running_row = []
            prr_denom = 0
            for a_mid in mids:
                prob_row.append(compute_class_model_probabilities[a_mid])
                if self.sched_data[a_ccid][a_mid]['num_runs'] > 0:
                    # If it has been run here but not on other ccs, say this one is slow to motivate
                    # putting it elsewhere.
                    if a_mid in have_not_run_on_all_ccs:
                        avg_row.append(2.0)
                    else:
                        # Max_avg_runtime is simply used to make the values less extreme.
                        avg_row.append(self.sched_data[a_ccid][a_mid]['avg_runtime'] / float(max_avg_runtime))
                else:
                    if min_avg_runtime is None:
                        avg_row.append(1.0)
                    else:
                        # Make it believe it's slow so it gets more time to meet demands.
                        avg_row.append(1.0)
                job_per_time_row.append(1.0 / avg_row[-1])
                prr_denom += prob_row[-1] * avg_row[-1]
            for i in range(num_models):
                percent_running_row.append((prob_row[i] * avg_row[i]) / prr_denom)
            global_prob_matrix.append(prob_row)
            global_avg_matrix.append(avg_row)
            global_job_per_time_matrix.append(job_per_time_row)
            global_percent_running_matrix.append(percent_running_row)
        
        # If we're just doing an exact copy of fake ccs, don't need to do any optimization.
        # Just make the modified probs the same as if we were using fake ccs.
        if fake_cc_use == 'Copy':
            for cc_idx in range(num_ccs):
                for m_idx in range(num_models):
                    prob = global_prob_matrix[cc_idx][m_idx]
                    self.ccs[ccids[cc_idx]].model_group.models[mids[m_idx]].modified_prob = prob
            return
            
        global_work_vector = []
        for m_idx in range(num_models):
            global_work_vector.append(0)
            for cc_idx in range(num_ccs):
                global_work_vector[m_idx] += global_percent_running_matrix[cc_idx][m_idx] *\
                                             global_job_per_time_matrix[cc_idx][m_idx]

        # print('Global work vector:')
        # self.pp.pprint(global_work_vector)

        if global_dist_target is not None:
            tot = 0.0
            for i in range(len(global_work_vector)):
                tot += global_work_vector[i]
            print('Modified global work vector:')
            for m_idx in range(num_models):
                global_work_vector[m_idx] = global_dist_target[mids[m_idx]] * tot
            self.pp.pprint(global_work_vector)

        # Step 2: Now, set up the first LP to solve for what we want the running percents to be

        # First LP: |Models|*|Compute Classes| + 1 (new values for percent running + optimized value C)
        # Conceptual ordering is first by model and then by compute class:
        #   e.g. (m1, cc1), (m1, cc2), ... (m2, cc1), ..., C

        vector_size = num_models * num_ccs + 1

        # All zeros except for the last entry
        c = np.array([0 if i + 1 < vector_size else -1.0 for i in range(vector_size)])

        # Make sure the sum = 1 for every set of probabilities/percents
        # And make sure each probability/percent >= 0
        percent_sum_rows = []
        percent_sum_values = []
        percent_geq_rows = []
        percent_geq_values = []
        for cc_idx in range(num_ccs):
            sum_row = [0 for i in range(vector_size)]
            for m_idx in range(num_models):
                vector_idx = m_idx * num_ccs + cc_idx
                sum_row[vector_idx] = 1.0
                percent_geq_rows.append([-1.0 if i == vector_idx else 0 for i in range(vector_size)])
                percent_geq_values.append(0.0)
            percent_sum_rows.append(sum_row)
            percent_sum_values.append(1.0)

        # Make sure the result work vector is a scaling of the old work vector
        # (Not sure whether or not to make this an == or a <= constraint.)
        # (Written so that it can be treated as either.)
        work_rows = []
        work_values = []
        for m_idx in range(num_models):
            work_row = [0 for i in range(vector_size)]
            for cc_idx in range(num_ccs):
                vector_idx = m_idx * num_ccs + cc_idx
                work_row[vector_idx] = -1.0 * global_job_per_time_matrix[cc_idx][m_idx]
            work_row[-1] = global_work_vector[m_idx]
            work_rows.append(work_row)
            work_values.append(0.0)

        equality_constraints = percent_sum_rows + work_rows
        equality_values = percent_sum_values + work_values
        leq_constraints = percent_geq_rows
        leq_values = percent_geq_values

        result = linprog(c, A_ub=leq_constraints, b_ub=leq_values, A_eq=equality_constraints, b_eq=equality_values)
        if not result['success']:
            print('Failed to solve linear program 1!')
            return
        print('Expected speedup of ' + str(result['x'][-1]))

        # Now that we know the target percent runtime values, we can compute the percent assignment values.
        # No optimization of anything -- just a solution is desired.
        # In this computation the elements are first ordered by compute class, then by model.

        target_percent_runtime_values = result['x']

        vector_size = num_ccs * num_models
        c = [0.0 for i in range(vector_size)]

        work_rows = []
        work_values = []
        for cc_idx in range(num_ccs):
            for m_idx in range(num_models):
                work_row = [0 for i in range(vector_size)]
                vector_idx = cc_idx * num_models + m_idx
                old_vector_idx = m_idx * num_ccs + cc_idx
                target_value = 0
                if target_percent_runtime_values[old_vector_idx] >= 0.01:  # Throw away miniscule targets
                    target_value = target_percent_runtime_values[old_vector_idx]
                for m_idx_prime in range(num_models):
                    vector_idx_prime = cc_idx * num_models + m_idx_prime
                    work_row[vector_idx_prime] += target_value * global_avg_matrix[cc_idx][m_idx_prime]
                work_row[vector_idx] += -1.0 * global_avg_matrix[cc_idx][m_idx]
                work_rows.append(work_row)
                work_values.append(0)

        probability_sum_rows = []
        probability_sum_values = []
        probability_geq_rows = []
        probability_geq_values = []
        for cc_idx in range(num_ccs):
            sum_row = [0 for i in range(vector_size)]
            for m_idx in range(num_models):
                vector_idx = cc_idx * num_models + m_idx
                sum_row[vector_idx] = 1.0
                probability_geq_rows.append([-1.0 if i == vector_idx else 0 for i in range(vector_size)])
                probability_geq_values.append(0.0)
            probability_sum_rows.append(sum_row)
            probability_sum_values.append(1.0)

        equality_constraints = probability_sum_rows
        equality_values = probability_sum_values
        leq_constraints = probability_geq_rows + work_rows
        leq_values = probability_geq_values + work_values
        result = linprog(c, A_ub=leq_constraints, b_ub=leq_values, A_eq=equality_constraints, b_eq=equality_values)
        if not result['success']:
            print('Failed to solve linear program 2! Status: ' + str(result['status']))
            return
        # self.pp.pprint(result)

        # Send the modified probabilities to the models
        for cc_idx in range(num_ccs):
            for m_idx in range(num_models):
                vector_idx = cc_idx * num_models + m_idx
                prob = result['x'][vector_idx]
                if prob < -0.01:
                    print('Serious error! Negative Probability! ' + str(prob))
                    prob = 0.0
                elif prob < 0.0:
                    print('Minor error. Negative Residue. ' + str(prob))
                    prob = 0.0
                print('Assigning prob of ' + str(prob) + ' to model ' + mids[m_idx] + ' on ' + ccids[cc_idx])
                self.ccs[ccids[cc_idx]].model_group.models[mids[m_idx]].modified_prob = prob

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
        # print('############## Success! ##############')
        # print(json.dumps(results, indent=2, sort_keys=True))

        # Update the DB with the result
        self.ccs[ccid].register_result(model_id, result_id, loss, results)

        self.update_sched_data(ccid, model_id, results)
        self.modify_probabilities(fake_cc_use='Copy')

        # Reassign models to CCs at some frequency
        if self.backend.result_count % 10 == 0:
            self.assign_to_ccs()

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
