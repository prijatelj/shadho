import numpy as np

from perceptron import Perceptron

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
    # 4 model ids, 4 cc_ids, 3 random uniforms
    perceptron = Perceptron(11, 4, models, list(runtime_map.keys()))

    scheduler_state = {
        'a': np.random.choice(models, 2, False),
        'b': np.random.choice(models, 2, False),
        'c': np.random.choice(models, 2, False),
        'd': np.random.choice(models, 2, False)
    }
    print('initial scheduler_state = \n', scheduler_state)

    all_samples = []
    all_runtimes = []
    predictions = []

    samples = []
    runtimes = []
    # inital sample
    for i in range(100):
        cc_id = np.random.choice(list(runtime_map.keys()))
        model_id = np.random.choice(scheduler_state[cc_id])
        rand = np.random.uniform(size=3)
        samples.append([model_id, cc_id, rand[0], rand[1], rand[2]])
        runtimes.append(runtime_map[cc_id][model_id])

    perceptron.update(samples, runtimes)
    predictions.append(perceptron.predict(samples))

    print('scheduler_state = \n', predictions[-1][1][-1])
    print('prediction count = ', len(predictions), ' and ', len(predictions[0]))

    for s in range(10):
        if (10-s) < len(predictions[0][1]):
            print(-(10-s), 'sample and scheduler prediction: ')
            print(samples[-(10-s)])
            print(predictions[0][1][-(10-s)])

    """
    all_samples += samples
    all_runtimes += runtimes

    # repetitive update and predict iterations

    for update_predict_iter in range(1,100):
        samples = []
        runtimes = []

        for i in range(100):
            model_id = np.random.choice(model_ids)
            cc_id = np.random.choice(list(runtime_map.keys()))
            rand = np.random.uniform(size=3)
            samples.append([model_id, cc_id, rand[0], rand[1], rand[2]])
            runtimes.append(runtime_map[cc_id][model_id])

        perceptron.update(samples, runtimes)
        logits, schedule = predictions.append(perceptron.predict(samples))

        all_samples += samples
        all_runtimes += runtimes
    """
