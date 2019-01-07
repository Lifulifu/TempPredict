from modelGenerator import genModel
from data.dataGen import genXY
import random

'''
param = 
    kernelSize: 0
    convNeurons: (0,0,0)
    denseNeurons: (0,0,0)
    act: 'relu'
    lossFunc: 'mse'
'''

class World():
    
    def __init__(self, xtrain, ytrain, population, max_iter, param_choice, survival_rate, mutation_rate):
        self.xtrain = xtrain
        self.ytrain = ytrain
        self.maxIter = max_iter
        self.population = population #models per iteration
        self.paramChoice = param_choice
        self.currModels = []
        self.modelCount = 0
        self.survivalRate = survival_rate
        self.mutationRate = mutation_rate
        
        self.logLine('World init')
        
        
    def runWorld(self):
        self.firstGeneration()
        for generation in range(self.maxIter):
            self.logLine('Running generation ' + str(generation+1))
            self.runGeneration()
            self.logLine('Finishing generation ' + str(generation+1))
            self.updateGeneration()
        
        
    def firstGeneration(self):
        #create the first generation
        for i in range(self.population):
            self.currModels.append({
                'id': self.giveId(),
                'params': self.randomParam(),
                'score': 9999
            })          
        self.logLine('First generation created')
            
            
    def runGeneration(self):
        for m in self.currModels:
            hist = genModel('id{}'.format(m['id']), self.xtrain, self.ytrain,
                             m['params']['kernelSize'], m['params']['convNeurons'], m['params']['denseNeurons'],
                             m['params']['act'], m['params']['lossFunc'])
            m['score'] = min(hist.history['val_loss'])      
            self.logLine('model finished'+str(m))
           
        
    def updateGeneration(self):
        self.currModels.sort(key=lambda x: x['score'])
        
        #regular children
        newGeneration = []
        for i in range(self.population - self.mutationRate):
            parents = random.sample(self.currModels[:self.survivalRate], 2) #choose top models as parents
            child = self.mate(parents)
            child['id'] = self.giveId()
            child['score'] = 9999
            newGeneration.append(child)
            
        #mutate children
        for i in range(self.mutationRate):
            newGeneration.append({
                'id': self.giveId(),
                'params': self.randomParam(),
                'score': 9999
            })
        
        self.currModels = newGeneration
        self.logLine('New generation created')
        
        
    def mate(self, parents):
        p = parents.copy()
        for key, val in p[0]['params'].items():
            # swap both parents' param
            if random.random() < 0.5:
                temp = p[0]['params'][key]
                p[0]['params'][key] = p[1]['params'][key]
                p[1]['params'][key] = temp
        return p[0]
            
        
    def randomParam(self):
        result = self.paramChoice.copy()
        for key, val in self.paramChoice.items():
            result[key] = random.choice(val)
        return result       
        
        
    def giveId(self):
        self.modelCount = self.modelCount+1
        return '{:04d}'.format(self.modelCount)
    
    
    def logLine(self, string):
        with open('world_log.txt', 'a') as f:
            f.write(string+'\n')


paramChoice = {
    'kernelSize': [(i,i,i) for i in range(1, 22, 2)], #10*10*10*2*2
    'convNeurons': [(i,2*i,2*i) for i in range(1, 22, 2)],
    'denseNeurons': [(i+8,i+4,i) for i in range(6, 26, 2)],
    'act': ['relu', 'sigmoid', 'tanh'],
    'lossFunc': ['mse', 'mae']
}
trainx, trainy = genXY('new_data/2010_to_2018.csv', 72, 24)
w = World(trainx, trainy, 10, 10, paramChoice, 5, 2)
w.runWorld()
print(w.currModels)


    
    


