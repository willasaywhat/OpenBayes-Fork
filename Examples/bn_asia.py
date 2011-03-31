from OpenBayes import BNet, BVertex, DirEdge, JoinTree

network = BNet('Asia Bayesian Network')

# Create a discrete node for all nodes with 2 states
visit_to_asia = network.add_v(BVertex('Visit to Asia', True, 2))
smoking = network.add_v(BVertex('Smoking', True, 2))
tuberculosis = network.add_v(BVertex('Tuberculosis', True, 2))
lung_cancer = network.add_v(BVertex('Lung Cancer', True, 2))
bronchitis = network.add_v(BVertex('Bronchitis', True, 2))
tub_or_cancer = network.add_v(BVertex('Tuberculosis or Cancer', True, 2))
xray_result = network.add_v(BVertex('X-Ray Result', True, 2))
dyspnea = network.add_v(BVertex('Dyspnea', True, 2))

# Connect the nodes

# V -> T
network.add_e(DirEdge(len(network.e), visit_to_asia, tuberculosis))

# T -> TC
network.add_e(DirEdge(len(network.e), tuberculosis, tub_or_cancer))

# TC -> X
network.add_e(DirEdge(len(network.e), tub_or_cancer, xray_result))

# TC -> D
network.add_e(DirEdge(len(network.e), tub_or_cancer, dyspnea))

# S -> LC
network.add_e(DirEdge(len(network.e), smoking, lung_cancer))

# S -> B
network.add_e(DirEdge(len(network.e), smoking, bronchitis))

# B -> D
network.add_e(DirEdge(len(network.e), visit_to_asia, dyspnea))

# LC -> TC
network.add_e(DirEdge(len(network.e), lung_cancer, tub_or_cancer))

# Show the network
print network

# Initialize the distributions
network.InitDistributions()

# Set distributions for start nodes
visit_to_asia.setDistributionParameters([0.01, 0.99])
smoking.setDistributionParameters([0.5, 0.5])

# Tuberculosis 0 = true, 1 = false
tuberculosis.distribution[{'Visit to Asia':0}]=[0.35, 0.65]
tuberculosis.distribution[{'Visit to Asia':1}]=[0.01, 0.99]

# Lung Cancer
lung_cancer.distribution[{'Smoking':0}]=[0.7, 0.3] #Smoker
lung_cancer.distribution[{'Smoking':1}]=[0.2, 0.8] #NonSmoker

# Bronchitis
bronchitis.distribution[{'Smoking':0}]=[0.8, 0.2]
bronchitis.distribution[{'Smoking':1}]=[0.6, 0.4]

# Tuberculosis or Cancer
tub_or_cancer.distribution[{'Tuberculosis':0, 'Cancer':0}] = [0, 1]
tub_or_cancer.distribution[{'Tuberculosis':0, 'Cancer':1}] = [1, 0]
tub_or_cancer.distribution[{'Tuberculosis':1, 'Cancer':0}] = [1, 0]
tub_or_cancer.distribution[{'Tuberculosis':1, 'Cancer':1}] = [1, 0]

# X-Ray Result
xray_result.distribution[{'Tuberculosis or Cancer':0}] = [0.1, 0.9]
xray_result.distribution[{'Tuberculosis or Cancer':1}] = [0.5, 0.5]

# Dyspnea
dyspnea.distribution[{'Tuberculosis or Cancer':0, 'Bronchitis':0}] = [0, 1]
dyspnea.distribution[{'Tuberculosis or Cancer':0, 'Bronchitis':1}] = [0.4, 0.6]
dyspnea.distribution[{'Tuberculosis or Cancer':1, 'Bronchitis':0}] = [0.6, 0.4]
dyspnea.distribution[{'Tuberculosis or Cancer':1, 'Bronchitis':1}] = [0.8, 0.2]

# Build a JoinTree
join_tree = JoinTree(network)

###
## Begin example inferences
###

###
## Probability of Lung Cancer given Bronchitis and Smoking
###

# Give the network some evidence.
join_tree.SetObs({'Bronchitis':1, 'Smoking':1})

# Assume we know the patient went to asia, and has bronchitis
print "Probability of Lung Cancer given Bronchitis and Smoking: "

# Infer probability of Lung Cancer
print join_tree.Marginalise('Lung Cancer')


###
## Probability of Abnormal XRay given Smoker
###

join_tree.SetObs({'Smoking':1})
print "Probability of an Abnormal XRay given Smoking: "
print join_tree.Marginalise('X-Ray Result')

###
## Probability of Dyspnea given a Visit to Asia
###

join_tree.SetObs({'Visit to Asia': 1})
print "Probability of Dyspnea given a Visit to Asia: "
print join_tree.Marginalise('Dyspnea')
