from sklearn.model_selection import train_test_split
from utils import *
from RISE import RISE

RULE_FILENAME = "results/"
path = "datasets/"

# User selection of dataset
dataset_dict = {'1':'Heart-C (small)', '2':'Pima Diabetes (medium)', '3':'Rice (large)', '4':'Other datasets', '5':'Exit'}
selection = make_selection('Datasets', dataset_dict)
dataset_selected = dataset_dict[selection]

print('\nSelected ' + selection + ' - ' + str(dataset_selected) + '\n')

filepath_correct = True
while filepath_correct:
    if selection == "1":
        filepath = path + "heart-c.arff"
        RULE_FILENAME = RULE_FILENAME + "RULES_HEART.txt"
    elif selection == "2":
        filepath = path + "pima_diabetes.arff"
        RULE_FILENAME = RULE_FILENAME + "RULES_PIMA.txt"
    elif selection == "3":
        filepath = path + "rice.arff"
        RULE_FILENAME = RULE_FILENAME + "RULES_RICE.txt"
    elif selection == "4":
        input_filepath = input("Introduce the filename of your dataset"  + ': ')
        filepath = path + input_filepath
        RULE_FILENAME = RULE_FILENAME + "RULES_X.txt"
    elif selection == "5":
        exit()

    # Load chosen dataset
    flag, dataset = load_data(filepath)
    if flag == True:
        filepath_correct = False
    else:
        filepath_correct = True

# If the algorithm is "Rice", reduce number of instances
if selection == "3":
    dataset = dataset.sample(2500)
    dataset.reset_index(drop = True,inplace = True)

# Train/Test split
dx = dataset.drop(dataset.columns[-1], axis=1)
X_train, X_test, Y_train, Y_test = train_test_split(dx, dataset[dataset.columns[-1]], shuffle = True, test_size=0.3)

# RISE Algorithm
# Create instance of the RISE class
rise = RISE()
rise.RULE_FILENAME = RULE_FILENAME
# Calling the fit method
rise.fit(X_train, Y_train)
# Calling the test method
acc_test = rise.test(X_test, Y_test)
print("Test accuracy is: " + str(acc_test))

