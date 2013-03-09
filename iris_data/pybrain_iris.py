from pybrain.supervised.trainers import BackpropTrainer
from pybrain.datasets import ClassificationDataSet
from pybrain.tools.shortcuts import buildNetwork
from pybrain.utilities import percentError

from sklearn.datasets import load_iris

iris = load_iris()
X = iris['data']
Y = iris['target']

dataset = ClassificationDataSet(4,1, class_labels=iris['target_names'])
for x,y in zip(X, Y):
    dataset.addSample(x, y)

testData, trainData = dataset.splitWithProportion( 0.25 )
testData._convertToOneOfMany()
trainData._convertToOneOfMany()

print "Number of training samples: ", len (trainData)
print "Input / Output dimensions: ", trainData.indim, trainData.outdim
print "First Sample (input output class):"
print trainData['input'][0], trainData['target'][0], trainData['class'][0]

net = buildNetwork(4, 12, 3)
trainer = BackpropTrainer(net,dataset=trainData, momentum=0.1, weightdecay=0.01)
for i in range(150):
    trainer.trainEpochs(1)

    trainResult = percentError( trainer.testOnClassData(), trainData['class'] )
    testResult = percentError( trainer.testOnClassData( dataset = testData ), testData['class'] )

    print "Epoch %4d" % trainer.totalepochs, \
          "  Train Error: %5.2f%%" % trainResult, \
          "  Test Error:  %5.2f%%" % testResult

