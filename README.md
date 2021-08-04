# DeepSuperLearner (2018) in Python
This is a [sklearn](http://scikit-learn.org/stable/)
implementation of the machine-learning DeepSuperLearner algorithm, A Deep Ensemble method for Classification Problems.

For details about DeepSuperLearner please refer to the [https://arxiv.org/abs/1803.02323](https://arxiv.org/abs/1803.02323):
Deep Super Learner: A Deep Ensemble for Classification Problems by Steven Young, Tamer Abdou, and Ayse Bener.

### Installation and demo
1. Clone this repository
    ```bash
    git clone https://github.com/levyben/DeepSuperLearner.git
    ```

2. Install the python library
    ```bash
    cd DeepSuperLearner
    python setup.py install
    ```

### Example:
```python
    ERT_learner = ExtremeRandomizedTrees(n_estimators=200, max_depth=None, max_features=1)
    kNN_learner = kNearestNeighbors(n_neighbors=11)
    LR_learner = LogisticRegression()
    RFC_learner = RandomForestClassifier(n_estimators=200, max_depth=None)
    XGB_learner = XGBClassifier(n_estimators=200, max_depth=3, learning_rate=1.)
    Base_learners = {'ExtremeRandomizedTrees':ERT_learner, 'kNearestNeighbors':kNN_learner, 'LogisticRegression':LR_learner,
                     'RandomForestClassifier':RFC_learner, 'XGBClassifier':XGB_learner}    
    np.random.seed(100)
    X, y = datasets.make_classification(n_samples=1000, n_features=12,
                                        n_informative=2, n_redundant=6)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    DSL_learner = DeepSuperLearner(Base_learners)
    DSL_learner.fit(X_train, y_train)
    DSL_learner.get_precision_recall(X_test, y_test, show_graphs=True)    
```

See deepSuperLearner/example.py for full example.

![Alt text](example.PNG?raw=true "Example results on sklearn simple binary classification data")


### Notes: 
1. For running example you need to install the XGB python lib as it is used as a base learner just as done in the paper.
2. Although the algorithm is implemented for classification problems, it can be modified to perform on regression problems aswell.

### TODO:
- [x] Train on some sklearn data.
- [ ] Restore paper classification results.

