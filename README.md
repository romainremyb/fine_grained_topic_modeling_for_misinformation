# fine_grained_topic_modeling_for_misinformation Project

The libraries used for this project are listed in the [requirements.txt](requirements.txt) file. The experiment result files for dataset 1 and 2 are available [here](experiment_results/).

The [src](src) folder contains the source code of this project. In fact you will find: 
    - preprocessing methods inside this [file](src/utils.py).
    - scripts for constructing dataset [1](src/dataset1_to_csv.py) and [2](src/dataset2_to_csv.py)
    - Cimple KG handling methods are available [here](src/cimple_querying.py)
    - [this folder](src/tomodapi/) contains the wrappers from the TOMODAPI library
    - The [models folder](src/models/) contains the wrappers to the systems that I used for this project

Finally the [tests folder](tests/) regroups some files to vizualise the experiments or systems:
    - [here](tests/experiment_analysis.ipynb) is the file for vizualizing the experiment analysis for the models with respect to the three performance metrics
    - [this file](tests/bertopic_report.ipynb) is the one that I used for most BERTopics analysis. Note that the file has been re-ran since the writing of the report and you may find some slight differences in the results (contains, standard ad custom fitting + remodelling outlier label -> topic -1)
    - you can find some experimentations that I did with the seven input labels [here](tests/bertopic_seven_topics.ipynb)
    - the experiments that were done with the topic retriever and presented in the report is in [this](tests/bertopic_topicfinder_distances.ipynb) (again this file has been re-ran)
    - the metrics that I retrieved on the Cimple metrics have been generated from [this file](tests/extract_cimple_metrics.py) 



