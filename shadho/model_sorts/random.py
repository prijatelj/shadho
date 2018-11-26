"""
Random sorts for mapping models to compute classes
"""
import numpy as np

def uniform(model_ids, compute_class_ids, min_model_assignment=2):
    """
    Uniform random assignment. Every compute class has at least 1 model.

    :param model_ids: list of model_ids
    :param compute_class_ids: list of compute_class_ids
    :param min_model_assignment: The minimum number of compute classes each model is assigned to.
    """
    # Assign compute classes to every model, w/o replacement
    model_id_to_compute_classes = {model_id:np.random.choice(compute_class_ids, min_model_assignment, replace=False) for model_id in model_ids}

    #inverse the dictionary into compute_class to models, as SHADHO expects
    compute_class_to_models = {}
    for model_id, compute_classes in model_id_to_compute_classes.items():
        for compute_class_id in compute_classes:
            if compute_class_id not in compute_class_to_models:
                compute_class_to_models[compute_class_id] = [model_id]
            else:
                compute_class_to_models[compute_class_id].append(model_id)

    # NOTE not pure random, may want to remove this line.
    # if any compute_class_id missing models, then assign it 1: otherwise idle
    for compute_class_id in compute_class_ids:
        if compute_class_id not in compute_class_to_models:
            compute_class_to_models[compute_class_id] = [np.random.choice(model_ids)]

    return compute_class_to_models
