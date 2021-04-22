# RISE_Algorithm
Implementation of RISE algorithm

Inductive learning is a learning paradigm based on creating data representations by observing examples, either implicitly or explicitly. Instance-based learning and rule induction are two
popular inductive learning approaches that will be the foundation of this project. Whereas instance-based learning is based on finding the nearest example according to some similarity
metric, the purpose of rule induction is generating rules that represent a class definition in order to cover many positive classes.

By themselves, these two approaches have some flaws. However, their strengths and weaknesses are complementary, which has lead to usually combine them into multi-strategy learning approaches. One of the algorithms constructed in
this basis is the Rule Induction from a Set of Exemplars (RISE) algorithm [1]

[1] P. Domingos, "Unifying instance-based and rule-based induction."

---------------------------------------------------------------------

This project has been developed using Python v3.6 as programming language and PyCharm as
IDE. In order to execute the project, simply run "main.py" in PyCharm or some similar IDE.
You will be greeted with a menu that will provide you the opportunity of selecting between
several options:

• "1": Processing Heart-C dataset.
• "2": Processing Pima Diabetes dataset.
• "3": Processing Rice dataset.
• "4": Process another dataset.
• "5": Exit.

Before anything else, note that the PyCharm project is composed by three .py files and two
folders: "datasets" and "results". In order for the code to work properly, the dataset files should
be located inside a folder called "datasets" within the root path. The folder called "results" will
be created automatically.

By selecting "1", "2" or "3" you will start the processing of the three datasets used for testing the implementation, whereas by selecting option number "4" you will be able to process another
dataset of your choice. For that matter, you will have to provide the filename of your dataset, and verify that said file is located inside a folder "datasets" as just mentioned. In addition, only
.csv or .arff files will be properly loaded. Last option is just exiting the execution.

Select a studied dataset and some basic relevant information about it will be displayed in a
first instance. Afterwards, the initial rule set will be printed and the training will start. Each
time that an accuracy improvement is carried out, it will be showed in the screen. When the
training is over, the final training accuracy, the training time, the final number of rules and
the name of the .txt file containing them will be displayed. Moreover, the test phase will start
automatically and the test accuracy will be shown when the execution is completed.
Note that all .txt files will be saved in the "results" folder of the project.

