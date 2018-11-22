"""
Implementation of dynamic sort methods. The majority of these will rely on the
low-end Tensorflow API to have finer-grained control over the math.

:author: Derek S. Prijatelj
"""

def online_reinforcement_svm(shadho):
    #TODO extact the sample data, their loss params (runtime or throughput), labels = compute classes each model assigned to in order of model_id. Assumes model_id static throughout SHADHO run.

    # results will need to be pulled from every model instance:
    #shadho.ccs[cc_id_dict_keys].modelgroup.models[iterate_model_list].results
    # NOTE jeff states that shadho.backend has the single instance of all models and is always up-to-date because the model group models are references to these

    # Sample = all models and their corresponding resource metrics (also possibly their hyperparameters), their runtime, and each model's list of compute classes they were assigned to.

    # batch = all samples run in the time since last update. Affected by shadho.update_frequency.

    # incremental train

    # prediction made from udpated model

    #return the ccs to model id dict or just reassign here?
    return
