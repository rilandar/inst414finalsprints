INST414_ML_project
==============================

Using Random Forest and Gradient Boosting Regressor models to train and predict data from 2018-2020 and evaluate how well the models would work.

EnergyML
------------

    ├── README.md          <- TDescriped project, methodlogy and set up instructions.
    ├── requirements.txt            <- Include packages like pandas, sklearn, matplotlib, etc.
    ├── setup.py           <- Keep default
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Partially transformed data, picking out       specific factors, including Year, Revenue, Avg Price (cents/kWh), & Total retail sales (MWh).
    │   ├── processed      <- The final, canonical data sets for modeling has specified data with specific factors including: Year, Revenue, Avg Price (cents/kWh), & Total retail sales (MWh)
    │   └── raw            <- Raw data from U.S. EIA ranging from 1990-2023.
    │
    ├── docs               <- none needed
    │
    ├── models             <- Trained and serialized Trained Random Forest and Gradient Boosting models, model predictions.
    │
    ├── notebooks
    ├── INST414MLfinal.ipynb.py <- Training and evaluating models in Jupyter,Final graphs + evaluation
    │                         RI, ML learning model for energy consumption
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for 
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results     └── oriented visualizations 


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
Sure! Here's your `README.md` written in paragraph form without using dashes or bullet points:

---

# EnergyML

EnergyML is a machine learning project designed to predict energy consumption and revenue trends for the state of Maryland. The project uses historical data provided by the U.S. Energy Information Administration (EIA), covering the years 2018 to 2023, for a smaller dataset. The main goal is to forecast key variables such as revenue, average price, and total retail sales in megawatt-hours. To accomplish this, the project implements supervised learning models including Random Forest and Gradient Boosting Regressors.

The structure of the project follows the standard cookiecutter data science format to ensure it is organized, reproducible, and scalable. The data directory is separated into four categories: raw data contains the untouched EIA datasets, interim data holds partially cleaned data with selected features such as year and revenue, processed data includes the finalized datasets used for model training, and external data contains any third-party sources. Model outputs, including trained models and predictions, are stored in the models folder. Graphs, charts, and visualizations are placed in the reports/figures directory, and explanatory materials such as data dictionaries are available in the references folder.

Model development, training, and evaluation are performed in a Jupyter notebook titled INST414MLfinal.ipynb.py, located in the notebooks folder. The source code supporting the pipeline is organized under the src directory, which includes scripts for dataset creation, feature engineering, model training and prediction, and visualizations. Dependencies required to run the project are listed in the requirements.txt file, and the setup.py file allows the project to be installed as a package if needed.

To run the project, begin by cloning the repository to your local machine. After cloning, make sure you are in the project root directory. Install the required dependencies using the command pip install -r requirements.txt. Once everything is installed, you can open the notebook using JupyterLab or Jupyter Notebook and execute the cells in the correct order to preprocess data, train models, and generate visualizations. The project was designed using Python and libraries such as pandas, scikit-learn, and matplotlib. When importing the CSV data, make sure to replace the file path with the location of the cleaned data file, titled 'mllearningdata.csv', in the Cookiecutter project.

This project was created for an academic machine learning course and serves as a practical application of predictive modeling techniques in the context of energy planning and data-driven decision-making.
