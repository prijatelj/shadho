# -*- coding: utf-8 -*-
"""Main driver for distributed hyperparameter search.

Classes
-------
HyperparameterSearch: perform hyperparameter search with or without decisions
"""

from .forest import OrderedSearchForest
from .config import SHADHOConfig

import json
import os
import tarfile
import time
import uuid

import numpy as np

import work_queue
from work_queue import WorkQueue, Task, WORK_QUEUE_RESULT_SUCCESS


class SHADHO(object):
    """Manager for

    """
    def __init__(self, spec, cmd=None, ccs=None, infiles=None,
                 outfile='output.json', use_complexity=True, use_priority=True,
                 timeout=3600, max_tasks=500, threshold=-1):
        self.cfg = SHADHOConfig()
        self.wq = WorkQueue(port=int(self.config.workqueue['port']),
                            name=str(self.config.workqueue['name']),
                            catalog=self.config.workqueue['catalog'],
                            exclusive=self.config.workqueue['exclusive'],
                            shutdown=self.config.workqueue['shutdown']
                            )

        self.timeout = timeout
        self.threshold = threshold

        if ccs is None:
            self.search = HeuristicSearch(spec,
                                          use_complexity=use_complexity,
                                          use_priority=use_priority
                                          )
        else:
            self.search = HardwareAwareSearch(spec,
                                              ccs,
                                              use_complexity=use_complexity,
                                              use_priority=use_priority
                                              )

    def run(self):
        """Conduct the hyperparameter search.
        """
        thresh = False
        start = time.time()
        curr = 0
        while curr < timeout and not thresh:
            tasks = self.search.make_tasks()
            for task in tasks:
                self.wq.submit(task)
            res = self.wq.wait(timeout=10)
            if res is not None and res.result == WORK_QUEUE_RESULT_SUCCESS:
                self.__success(res)
            else:
                self.__failure(res)

            curr = time.time() - start
            if self.search.min < self.threshold:
                thresh = True

        self.search.report_min()


class HeuristicSearch(object):
    """Randomly generate hyperparameter values.

    Parameters
    ----------
    spec : dict
        Specification tree for the hyperparameter search.
    use_complexity : {True, False}
        True to weight parameter generation by the complexity heuristic.
    use_priority : {True, False}
        True to weight parameter generation by the priority heuristic.


    Attributes
    ----------
    forest : shadho.forest.OrderedSearchForest
        Forest of potential hyperparameter values.


    """

    def __init__(self, spec, cc=None, use_complexity=True, use_priority=True,
                 max_tasks=500):
        self.forest = OrderedSearchForest(spec)
        self.cc = cc if cc is not None \
                  else ComputeClass('dummy', randint(0, len(self.forest)))
        self.use_complexity = use_complexity
        self.use_priority = use_priority
        self.current_tasks = 0
        self.max_tasks = max_tasks

    def get_params(self, n):
        """Generate hyperparameter values to test.

        Parameters
        ----------
        n : int
            The number of parameters to generate.

        Returns
        -------
        params : list(dict)
            A list of hyperparameter values to test.

        """
        self.forest.set_ranks(use_complexity=self.use_complexity,
                              use_priority=self.use_priority)
        return [self.forest.generate(self.cc.rv,
                                     use_complexity=self.use_complexity,
                                     use_priority=self.use_priority)
                for _ in range(n)]

    def make_tasks(self):
        params = self.get_params(self.max_tasks - self.current_tasks)



class HyperparameterSearch(object):
    """Perform distributed hyperparameter search.

    This class conducts and manages distributed hyperparameter search using the
    data structures, heuristics, and techniques described in REFERENCE. The
    search is fully configurable, meaning users may define the expected runtime
    environments, whether to direct the search using complexity and priority
    heuristics, and stopping conditions.

    Parameters
    ----------
    spec : dict
        Specification to build an OrderedSearchForest.
    ccs : list(shadho.ComputeClass)
        The ordered list of compute classes that will connect.
    wq_config : {dict, shadho.WQConfig}
        Work Queue master and task configurations.
    use_complexity: {True, False}, optional
        If True, use the complexity heuristic to adjust search proportions.
    use_priority: {True, False}, optional
        If True, use the priority heuristic to adjust search proportions.
    timeout: {600, int}, optional
        Search timeout in seconds.
    max_tasks: {100, int}, optional
        The maximum number of concurrent searches to maintain.

    See Also
    --------

    """

    def __init__(self, spec, ccs, wq_config, use_complexity=True,
                 use_priority=True, timeout=600, max_tasks=100):
        self.wq_config = WQConfig(**wq_config) \
                            if isinstance(wq_config, dict) else wq_config
        work_queue.cctools_debug_flags_set("all")
        work_queue.cctools_debug_config_file(self.wq_config['debug'])
        work_queue.cctools_debug_config_file_size(0)
        self.wq = WorkQueue(port=int(self.wq_config['port']),
                            name=str(self.wq_config['name']),
                            catalog=self.wq_config['catalog'],
                            exclusive=self.wq_config['exclusive'],
                            shutdown=self.wq_config['shutdown']
                            )
        self.forest = OrderedSearchForest(spec)
        self.ccs = [] if ccs is None else ccs
        self.use_complexity = use_complexity
        self.use_priority = use_priority
        self.timeout = timeout
        self.max_tasks = max_tasks
        self.wq.specify_log(self.wq_config['logfile'])

    def optimize(self):
        """Run the distributed hyperparameter search.
        """
        start = time.time()
        elapsed = 0
        if len(self.ccs) > 0:
            self.assign_to_ccs()
            for cc in self.ccs:
                print(str(cc))
        while elapsed < self.timeout:
            n_tasks = self.max_tasks - self.wq.stats.tasks_waiting
            tasks = self.__generate_tasks(n_tasks)
            for task in tasks:
                self.wq.submit(task)
            task = self.wq.wait(timeout=30)
            if task is not None:
                if task.result == work_queue.WORK_QUEUE_RESULT_SUCCESS:
                    self.__success(task)
                else:
                    self.__failure(task)
            elapsed = time.time() - start
            if len(self.ccs) > 0:
                for cc in self.ccs:
                    print(str(cc))

        self.forest.write_all()

    def __generate_tasks(self, n_tasks):
        """Generate values to search.

        Parameters
        ----------
        n_tasks : int
            The number of tasks to generate.

        Returns
        -------
        tasks : list(work_queue.Task)
            The tasks to submit to Work Queue.
        """
        tasks = []
        if len(self.ccs) == 0:
            self.forest.set_ranks(use_priority=self.use_priority,
                                  use_complexity=self.use_complexity)
            for i in range(n_tasks):
                task = work_queue.Task(self.wq_config['command'])

                tid, spec = self.forest.generate()
                tag = '.'.join([str(uuid.uuid4()), tid])
                task.specify_tag(tag)

                for f in self.wq_config['files']:
                    f.add_to_task(task, tag=''
                                  if f.type == work_queue.WORK_QUEUE_INPUT
                                  else str(tag + '.'))

                task.specify_buffer(str(json.dumps(spec)),
                                    remote_name=str('hyperparameters.json'),
                                    flags=work_queue.WORK_QUEUE_NOCACHE
                                    )
                tasks.append(task)
        else:
            self.assign_to_ccs()
            for cc in self.ccs:
                specs = cc.generate()
                for s in specs:
                    tid, spec = s
                    task = work_queue.Task(self.wq_config['command'])

                    tag = '.'.join([str(uuid.uuid4()), cc.name, tid])
                    task.specify_tag(tag)
                    if cc.resource == 'cores':
                        task.specify_cores(cc.value)

                    for f in self.wq_config['files']:
                        f.add_to_task(task, tag=''
                                      if f.type == work_queue.WORK_QUEUE_INPUT
                                      else str(tag + '.'))

                    task.specify_buffer(str(json.dumps(spec)),
                                        remote_name=str('hyperparameters.json'),
                                        flags=work_queue.WORK_QUEUE_NOCACHE
                                        )
                    tasks.append(task)
        return tasks

    def assign_to_ccs(self):
        """Assign hyperparameter search spaces to compute classes.

        This function iterates over the set of trees five times and over the
        compute classes twice. If we can reduce this, that would be suuuuper.
        """
        # TODO: Reduce the number of times iterating over things
        # TODO: Make the assignments agnostic to the number of trees and ccs
        # Clear the assignments within both trees and Compute Classes
        for cc in self.ccs:
            cc.clear_assignments()

        self.forest.set_ranks(use_priority=self.use_priority,
                              use_complexity=self.use_complexity)
        trees = [self.forest.trees[key] for key in self.forest.trees]
        trees.sort(key=lambda x: x.rank)

        # larger, smaller = (trees, self.ccs) \
        #     if len(trees) > len(self.ccs) else (self.ccs, trees)
        x = float(len(larger)) / float(len(smaller))
        y = x - 1
        j = 0
        n = len(larger) / 2

        for i in range(len(larger)):
            if i > np.ceil(y):
                j += 1
                y += x
            larger[i].assign(smaller[j])
            smaller[j].assign(larger[i])

            if i <= n:
                larger[i].assign(smaller[j + 1])
                smaller[j + 1].assign(larger[i])

            if i > n:
                larger[i].assign(smaller[j - 1])
                smaller[j - 1].assign(larger[i])

    def __success(self, task):
        """Default handling for successful task completion.
        """
        # Task tag is unique and contains information about the tree its values
        # came from and the compute class it was assigned to.
        tag = str(task.tag)
        print('Task {} was successful'.format(tag))
        ids = tag.split('.')

        # Get the correct tree from the OSF
        tree_id = ids[-1]
        tree = self.forest.trees[tree_id]

        # Extract the results from the output tar file.
        try:
            result = tarfile.open('.'.join([tag, 'out.tar.gz']), 'r')
            resultstring = result.extractfile('performance.json').read()
            result.close()
        except IOError:
            print('Error opening task {} result'.format(tag))

        # Load the results from file and store them to the correct tree
        result = json.loads(resultstring.decode('utf-8'))
        result['task_id'] = task.id
        tree.add_result(result['params'],
                        result,
                        update_priority=self.use_priority)

        # If using Compute Classes, update task statistics.
        if len(self.ccs) > 0:
            ccid = ids[1]
            for cc in self.ccs:
                if cc.name == ccid:
                    cc.submitted_tasks -= 1

        # Clean up
        os.remove('.'.join([tag, 'out.tar.gz']))

    def __failure(self, task):
        """Default handling for task failure.
        """
        # Report the failure and print any output for debugging.
        print('Task {} failed with result {} and WQ status {}'
              .format(task.tag, task.result, task.return_status))
        print(task.output)

        # If using Compute Classes, update task statistics.
        if len(self.ccs) > 0:
            ccid = task.tag.split('.')[1]
            for cc in self.ccs:
                if cc.name == ccid:
                    cc.submitted_tasks -= 1
