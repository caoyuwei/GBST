# GBST

GBST is an optimized distributed gradient boosting survival trees library that is implemented based on the XGBoost(https://github.com/dmlc/xgboost). It is designed to be highly ***efficient***, ***flexible*** and ***portable***.
It implements Gradient Boosting Survival Trees algorithms from the paper [Gradient Boosting Survival Tree with Applications in Credit Scoring](https://arxiv.org/abs/1908.03385).
With XGBoost as its core, GBST inherits the parallel tree boosting (also known as GBDT/GBM) to solve the problem of survival analysis efficiently and accurately .

License
-------
© Contributors, 2019. Licensed under an [Apache-2](https://github.com/TheSignPainter/GBST/blob/master/LICENSE) license.




Installation
------------
- Requirements
    - (for python package) python3(>=3.5.0), numpy, scipy, scikit-learn, (optional) pandas, (optional) matplotlib.
    - (for compiling from source) All the above, plus CMake(>=3.12), and a C++ compiler supporting C++11(e.g. g++-5.0 or higher)

- Install the python package only
    We provide the compiled library file (for Win64 and Linux) within the python package. To install, simply execute 
    ```
    python setup.py install
    ```
    in the repository "./gbst_package".

- Compile from source
    The folder gbst_src contains the source codes to compile the library file. To build it from source, we start from making a working directory:
    ```
    cd gbst_src
    mkdir build
    ```
    Then, for Windows:
    ```
    cmake .. -G "Visual Studio 14 2015 Win64"
    # for VS15: cmake .. -G"Visual Studio 15 2017" -A x64
    # for VS16: cmake .. -G"Visual Studio 16 2019" -A x64
    cmake --build . --config Release
    ```
    If the build process successfully ends, you should find the library file **gbst.dll** in the ./gbst_src/lib folder.

    For linux:
    ```
    cmake .. 
    make -j$(nproc)
    ```
    If the build process successfully ends, you should find the library file **libgbst.so** in the ./gbst_src/lib folder.

    After that, please copy the compiled library into gbst_package/GBST/ , in place of the pre-compiled library file(s), then install the python package by previous instructions.

Usage
---------

As in XGBoost, GBST provides interfaces to scikit-learn. One can easily define and tune a Survival Tree model, save/load it via local storage devices, and infer the hazard functions by the model. 

- Interfaces to scikit-learn, inherit from sklearn::BaseEstimator
- Hyper parameter searching/cross validation, as is in sklearn
- Plotting the tree's structure in [Graphviz](http://www.graphviz.org/)
- Plotting the importance of each feature


Reference
---------
- Miaojun Bai, Yan Zheng, and Yun Shen. [Gradient Boosting Survival Tree with Applications in Credit Scoring](https://arxiv.org/abs/1908.03385). arXiv preprint arXiv:1908.03385 (2019).
- Tianqi Chen and Carlos Guestrin. [XGBoost: A Scalable Tree Boosting System](http://arxiv.org/abs/1603.02754). In 22nd SIGKDD Conference on Knowledge Discovery and Data Mining, 2016
