import numpy as np


from perceptron import Perceptron

def convert_predictions_to_schedule(model_to_compute_classes):
    """
    :param model_to_compute_class: list of single entry dicts
    :return: dict of compute classes to np.array of model ids
    """
    # NOTE unnecessary if I specify exactly which model to run where!
    # NOT FOR RUNTIMES, for updating the scheduler state.

    compute_class_to_models = {}
    for mapping in model_to_compute_classes:
        model_id = list(mapping.keys())[0]
        compute_classes = list(mapping.values())[0]
        for compute_class_id in compute_classes:
            if compute_class_id not in compute_class_to_models:
                compute_class_to_models[compute_class_id] = np.array([model_id])
            else:
                compute_class_to_models[compute_class_id] = np.append(compute_class_to_models[compute_class_id], model_id)
    return compute_class_to_models

def create_samples(scheduler_state):
    new_samples = []
    for cc_id in scheduler_state: # 16 = 4*4
        for model_id in scheduler_state[cc_id]:
            rand = np.random.uniform(size=3) # Noise vars
            new_samples.append([model_id, cc_id, rand[0], rand[1], rand[2]])

    return new_samples

def get_runtimes(predictions, runtime_map):
    new_runtimes = []
    for pred in predictions:
        model_id = list(pred.keys())[0]
        for cc_id in list(pred.values())[0]:
            new_runtimes.append(runtime_map[cc_id][model_id])
    return new_runtimes

if __name__ == '__main__':
    # 4 different compute classes (a,b,c,d)
    # 4 different models (w,x,y,z)
    # Every model a compute class it runs best on.
    # 1-to-1 ideal is a:w, b:x, c:y, d:z.
    # However, models are able to run on multiple compute classes.
    runtime_map= {
        'cc_1':{
            'model_1':1,
            'model_2':2,
            'model_3':3,
            'model_4':4
        },
        'cc_2':{
            'model_1':4,
            'model_2':1,
            'model_3':2,
            'model_4':3
        },
        'cc_3':{
            'model_1':3,
            'model_2':4,
            'model_3':1,
            'model_4':2
        },
        'cc_4':{
            'model_1':2,
            'model_2':3,
            'model_3':4,
            'model_4':1
        },
    }

    # ascending order of classes have worse performance with more models.
    #len(a) * 1
    #len(b_assigned_models) * 2
    #len(c_assigned_models) * 3
    #len(d_assigned_models) * 4

    models = ['model_1', 'model_2', 'model_3', 'model_4']
    compute_classes = ['cc_1', 'cc_2', 'cc_3', 'cc_4']

    # Init Perceptron: 4 model ids, 4 cc_ids, 3 random uniforms
    perceptron = Perceptron(11, 4, models, list(compute_classes))

    # TODO initialize first state, must run all models on each compute class
    scheduler_state = {
        'cc_1': np.array(models),
        'cc_2': np.array(models),
        'cc_3': np.array(models),
        'cc_4': np.array(models)
    }
    print('initial scheduler_state = \n', scheduler_state)

    all_samples = []
    all_runtimes = []
    predictions = []

    samples = []
    runtimes = []
    # create inital sample: for every cc, assign each model type.
    samples.extend(create_samples(scheduler_state))

    # get predictions' runtimes and update.
    predictions.append(perceptron.predict(samples))
    runtimes.extend(get_runtimes(predictions[0][1], runtime_map))
    perceptron.update(samples, runtimes)

    # inverse the dictionary into compute_class to models, as SHADHO expects
    # assumes that this includes all compute classes, if not, make point to empty
    pred_scheds = convert_predictions_to_schedule(predictions[0][1])
    scheduler_state.update(pred_scheds)
    for cc_id in compute_classes:
        if cc_id not in pred_scheds:
            scheduler_state[cc_id] = np.array([]) # NOTE perhaps this is a problem?

    print('scheduler_state = \n', predictions[-1][1][-1])

    # print out most recent 10 samples and their associated predictions
    for s in range(10):
        if (10-s) < len(predictions[0][1]):
            print(-(10-s), 'sample and scheduler prediction: ')
            print(samples[-(10-s)])
            print(predictions[0][1][-(10-s)])

    #"""
    all_samples += samples
    all_runtimes += runtimes

    #
    # repetitive update and predict iterations
    #

    for update_itr in range(1,10):
        samples = []
        runtimes = []

        # generate samples
        samples.extend(create_samples(scheduler_state))

        # get predictions' runtimes and update.
        predictions.append(perceptron.predict(samples))
        runtimes.extend(get_runtimes(predictions[update_itr][1], runtime_map))
        perceptron.update(samples, runtimes)

        pred_scheds = convert_predictions_to_schedule(predictions[update_itr][1])
        scheduler_state.update(pred_scheds)
        for cc_id in compute_classes:
            if cc_id not in pred_scheds:
                scheduler_state[cc_id] = np.array([])

        all_samples += samples
        all_runtimes += runtimes

        # print out most recent 10 samples and their associated predictions
        for s in range(10):
            if (10-s) < len(predictions[update_itr][1]):
                print(-(10-s), 'sample and scheduler prediction: ')
                print(samples[-(10-s)])
                print(predictions[update_itr][1][-(10-s)])
    #"""
