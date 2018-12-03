import json
import sys
import pprint

with open(sys.argv[1] + '/results.json', 'r') as f:
    results = json.load(f)
# print(json.dumps(results, indent=2, sort_keys=True))
minLoss = 10000
totalSuccesses = 0
for i in range(len(results)):
    successes = len([r for r in results[i]['results'] if r['loss'] is not None])
    print(results[i]['domains'][0]['domain'][0] + '\t\t' + str(successes))
    totalSuccesses += successes
    # print(json.dumps(results[i]['domains'], indent=2, sort_keys=True))
    for j in range(len(results[i]['results'])):
        currLoss = results[i]['results'][j]['loss']
        if currLoss is not None and currLoss < minLoss:
            minLoss = currLoss
print('total successes: ' + str(totalSuccesses))
print('min loss: ' + str(minLoss))
