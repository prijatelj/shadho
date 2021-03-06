"""Facilities for grouping distributed hardware and search spaces.

Classes
-------
ComputeClass
    Group hardware by a common property.
"""
import uuid
import sys

from pyrameter.modelgroup import ModelGroup
from pyrameter.models.model import Model


class ComputeClass(object):
    """Group hardware by a common property.

    When working with heterogeneous distributed hardware (e.g., multiple
    clusters, multiple cloud services), it is often useful to group machines by
    a common resource. ComputeClass allows SHADHO to target specific hardware
    for running specific tasks. Hardware may be grouped by:

        * CPU count
        * Available RAM
        * GPU model
        * GPU count
        * Other accelerators
        * and by arbitrary, user-defined values.

    Parameters
    ----------
    name : str
        User-level name for the compute class (e.g. "8-core", "1080ti", etc.).
    resource : str
        The name of the resource common to this group.
    value
        The value of the resource common to this group.
    max_tasks : int
        The maximum number of tasks to queue. Recommended to be 1.5-2x the
        number of expected nodes with this resource.

    Attributes
    ----------
    id : str
        Internal id of this ComputeClass.
    name : str
        User-level name for the compute class (e.g. "8-core", "1080ti", etc.).
    resource : str
        The name of the resource common to this group.
    value
        The value of the resource common to this group.
    max_tasks : int
        The maximum number of tasks to queue. Recommended to be 1.5-2x the
        number of expected nodes with this resource.
    current_tasks : int
        The current number of queued tasks.
    model_group : `pyrameter.ModelGroup`
        The (possibly ordered) list of models assigned to this ComputeClass.

    See Also
    --------
    `pyrameter.ModelGroup`
    """
    def __init__(self, name, resource, value, max_tasks):
        self.id = str(uuid.uuid4())
        self.name = name
        self.resource = resource
        self.value = value
        self.max_tasks = max_tasks
        self.current_tasks = 0

        self.model_group = ModelGroup()

    def __hash__(self):
        return hash((self.id, self.name, self.resource, self.value))

    def generate(self, model_id=None):
        """Generate a set of hyperparameters from a model in this group.

        Hyperparameters are generated from a single model assigned to this
        compute class. If no model id is provided, the model is selected based
        on a probability distribution over the models.

        Parameters
        ----------
        model_id : str, optional
            If a valid id is supplied, generate hyperparameters values from the
            requested model. Otherwise, probabilistically select a model and
            generte hyperparameter values.
        """
        return self.model_group.generate(model_id)

    def add_model(self, model):
        """Add a model to this compute class.

        Parameters
        ----------
        model : `pyrameter.Model`
        """
        self.model_group.add_model(model)

    def remove_model(self, model_id):
        """Remove a model from this compute class.

        Parameters
        ----------
        model_id : str
        """
        self.model_group.remove_model(model_id)

    def clear(self):
        """Remove all models from this compute class."""
        self.model_group.clear()

    def register_result(self, model_id, result_id, loss, results):
        """Add a result to a model in this compute class.

        Parameters
        ----------
        model_id : str
            The id of the model to store the result in.

        """
        if not isinstance(results, dict):
            results = {'results': results}
        results['compute_class'] = (self.resource, self.value)
        return self.model_group.register_result(model_id, result_id, loss,
                                                results)
