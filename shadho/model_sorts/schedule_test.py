import numpy as np
import argparse

from perceptron import Perceptron

def parse_args():
    parser = argparse.ArgumentParser(description='Simulation of simplified sorting problem to be solved by online reinforcement learning model.')

    parser.add_argument('-i', '--iterations', default=100, type=int, help='Number of update iterations to simulate for the scheduler being tested.')

    args = parser.parse_args()

    if args.iterations <=0:
        parser.error('iterations must be an integer > 0.')

    return args

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

def create_samples(scheduler_state, preds=None):
    new_samples = []
    if preds is None:
        for cc_id in scheduler_state: # 16 = 4*4
            for model_id in scheduler_state[cc_id]:
                rand = np.random.uniform(size=3) # Noise vars
                new_samples.append([model_id, cc_id, rand[0], rand[1], rand[2]])
    else: # simulate running exact prediction by adding instnace features
        for pred in preds: # ignores scheduler state
            model_id = pred[0]
            for cc_id in pred[1:]:
                rand = np.random.uniform(size=3)
                new_samples.append([model_id, cc_id, rand[0], rand[1], rand[2]])

    return new_samples

def get_runtimes(samples, runtime_map):
    new_runtimes = []
    for sample in samples:
        new_runtimes.append(runtime_map[sample[1]][sample[0]])
    return new_runtimes

def update_scheduler_state(predictions, scheduler_state):
    pred_scheds = convert_predictions_to_schedule(predictions)
    scheduler_state.update(pred_scheds)
    for cc_id in compute_classes:
        if cc_id not in pred_scheds:
            scheduler_state[cc_id] = np.array([]) # NOTE perhaps this is a problem?

    return scheduler_state

if __name__ == '__main__':
    args = parse_args()
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
            'model_4':4,
            #'model_5':5
        },
        'cc_2':{
            'model_1':4,
            'model_2':1,
            'model_3':2,
            'model_4':3,
            #'model_5':4
        },
        'cc_3':{
            'model_1':3,
            'model_2':4,
            'model_3':1,
            'model_4':2,
            #'model_5':3
        },
        'cc_4':{
            'model_1':2,
            'model_2':3,
            'model_3':4,
            'model_4':1,
            #'model_5':2
        },
    }

    # NOTE could increase runtime on machine w/ more models assigned to them
    models = ['model_1', 'model_2', 'model_3', 'model_4']
    #models = ['model_1', 'model_2', 'model_3', 'model_4', 'model_5']
    compute_classes = ['cc_1', 'cc_2', 'cc_3', 'cc_4']

    # Init Perceptron: # model ids, # cc_ids, 3 random uniforms
    perceptron = Perceptron(len(models) + len(compute_classes) + 3, len(compute_classes), models, compute_classes)

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

    # create inital sample: for every cc, assign each model type.
    # Initial Run: get initial samples and runtimes
    samples = create_samples(scheduler_state)
    runtimes = get_runtimes(samples, runtime_map)

    # update and predict
    perceptron.update(samples, runtimes)
    predictions.append(perceptron.predict(samples))

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

    for update_itr in range(1,args.iterations):
        # generate samples and their runtimes
        samples = create_samples(scheduler_state, predictions[update_itr-1][1])
        runtimes = get_runtimes(samples, runtime_map)

        # update and predict
        perceptron.update(samples, runtimes)
        predictions.append(perceptron.predict(samples))

        # save all samples and runtimes for future reference
        all_samples += samples
        all_runtimes += runtimes

        # print out most recent 10 samples and their associated predictions
        for s in range(10):
            if (10-s) < len(predictions[update_itr][1]):
                print(-(10-s), 'sample and scheduler prediction: ')
                print(samples[-(10-s)])
                print(predictions[update_itr][1][-(10-s)])
    #"""
