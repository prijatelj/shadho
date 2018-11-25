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
    perceptron = Perceptron(8, models, list(runtime_map.keys()))

    samples = []
    runtimes = []
    for i in range(100):
        model_id = np.random.choice(model_ids)
        cc_id = np.random.choice(list(runtime_map.keys()))
        rand = np.random.uniform(size=3)
        samples.append([model_id, cc_id, rand[0], rand[1], rand[2]])
        runtimes.append(runtime_map[cc_id][model_id])

    perceptron.update(samples, runtimes)
    perceptron.predict(samples)