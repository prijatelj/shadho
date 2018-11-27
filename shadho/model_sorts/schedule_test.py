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

if __name__ == '__main__':
    # 4 different compute classes (a,b,c,d)
    # 4 different models (w,x,y,z)
    # Every model a compute class it runs best on.
    # 1-to-1 ideal is a:w, b:x, c:y, d:z.
    # However, models are able to run on multiple compute classes.
    runtime_map= {
        'a':{
            'w':1,
            'x':2,
            'y':3,
            'z':4
        },
        'b':{
            'w':4,
            'x':1,
            'y':2,
            'z':3
        },
        'c':{
            'w':3,
            'x':4,
            'y':1,
            'z':2
        },
        'd':{
            'w':2,
            'x':3,
            'y':4,
            'z':1
        },
    }

    # ascending order of classes have worse performance with more models.
    #len(a) * 1
    #len(b_assigned_models) * 2
    #len(c_assigned_models) * 3
    #len(d_assigned_models) * 4

    models = ['w', 'x', 'y', 'z']
    compute_classes = ['a', 'b', 'c', 'd']

    # 4 model ids, 4 cc_ids, 3 random uniforms
    perceptron = Perceptron(11, 4, models, list(compute_classes))

    # TODO initialize first state, must run all models on each compute class
    scheduler_state = {
        'a': np.array(models),
        'b': np.array(models),
        'c': np.array(models),
        'd': np.array(models)
        #'a': np.random.choice(models, 2, False),
        #'b': np.random.choice(models, 2, False),
        #'c': np.random.choice(models, 2, False),
        #'d': np.random.choice(models, 2, False)
    }
    print('initial scheduler_state = \n', scheduler_state)

    all_samples = []
    all_runtimes = []
    predictions = []

    samples = []
    runtimes = []
    # inital sample
    #for i in range(100):
    for cc_id in scheduler_state: # 16 = 4*4
        #cc_id = np.random.choice(list(compute_classes))
        #model_id = np.random.choice(scheduler_state[cc_id])
        for model_id in scheduler_state[cc_id]:
            rand = np.random.uniform(size=3)
            samples.append([model_id, cc_id, rand[0], rand[1], rand[2]])

    # get predictions' runtimes and update.
    predictions.append(perceptron.predict(samples))
    for pred in predictions[0][1]:
        model_id = list(pred.keys())[0]
        for cc_id in list(pred.values())[0]:
            runtimes.append(runtime_map[cc_id][model_id])
    perceptron.update(samples, runtimes)

    # TODO update the scheduler_state
    #for model in models:
    #inverse the dictionary into compute_class to models, as SHADHO expects
    # assumes that this includes all compute classes, if not, make point to empty
    pred_scheds = convert_predictions_to_schedule(predictions[0][1])
    scheduler_state.update(pred_scheds)
    for cc_id in compute_classes:
        if cc_id not in pred_scheds:
            scheduler_state[cc_id] = np.array([])


    #print('scheduler_state = \n', predictions[-1][1])
    print('scheduler_state = \n', predictions[-1][1][-1])
    #print('prediction count = ', len(predictions), ' and ', len(predictions[0]))

    for s in range(10):
        if (10-s) < len(predictions[0][1]):
            print(-(10-s), 'sample and scheduler prediction: ')
            print(samples[-(10-s)])
            print(predictions[0][1][-(10-s)])

    #"""
    all_samples += samples
    all_runtimes += runtimes

    # repetitive update and predict iterations

    for update_predict_iter in range(1,100):
        samples = []
        runtimes = []

        # generate samples
        #for i in range(100):
        #    model_id = np.random.choice(models)
        #    cc_id = np.random.choice(compute_classes)
        #    rand = np.random.uniform(size=3)
        #    samples.append([model_id, cc_id, rand[0], rand[1], rand[2]])
        for cc_id in scheduler_state: # 16 = 4*4
            for model_id in scheduler_state[cc_id]:
                rand = np.random.uniform(size=3)
                samples.append([model_id, cc_id, rand[0], rand[1], rand[2]])

        # get predictions' runtimes and update.
        predictions.append(perceptron.predict(samples))
        for pred in predictions[update_predict_iter][1]:
            model_id = list(pred.keys())[0]
            for cc_id in list(pred.values())[0]:
                runtimes.append(runtime_map[cc_id][model_id])
        perceptron.update(samples, runtimes)

        pred_scheds = convert_predictions_to_schedule(predictions[update_predict_iter][1])
        scheduler_state.update(pred_scheds)
        for cc_id in compute_classes:
            if cc_id not in pred_scheds:
                scheduler_state[cc_id] = np.array([])

        all_samples += samples
        all_runtimes += runtimes


        for s in range(10):
            if (10-s) < len(predictions[update_predict_iter][1]):
                print(-(10-s), 'sample and scheduler prediction: ')
                print(samples[-(10-s)])
                print(predictions[update_predict_iter][1][-(10-s)])
    #"""
