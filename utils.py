from scipy.io import arff
import pandas as pd
import numpy as np
import os

# Load and examine dataset
def load_data(filepath):
    if ".csv" in filepath:
        data = pd.read_csv(filepath)
    elif ".arff" in filepath:
        data, _ = arff.loadarff(filepath)
    else:
        print("File not found.\n")
        return False, 0
    df = pd.DataFrame(data)
    # Dataset dimension
    print("Dimensions of the dataset: " + str(df.shape))
    # There are 1473 instances with 9 attributes each and the corresponding class.

    print("\nInformation about the attributes:")
    print(df.info())
    # This dataset has numerical and categorical attributes but the advantage
    # is that nominal attributes are numerically labelled so all the dataset
    # can be transform into a numerical dataset.
    # Moreover, there aren't missing values

    # Peek at the data
    print("\nFirst 5 rows of the dataset:")
    print(df.head(5))

    print("\nStatistical summary:")
    print(df.describe())
    # The values of the numeral attributes has different ranges. This indicates that a normalization or
    # standardization will be necessary.

    print("\nClasses: " + str(df[df.columns[-1]].unique()))
    print()

    return True, df

# Preprocess dataset
def preprocessing(dataset):
    for attr in dataset.columns:
        if is_categorical(dataset[attr]):
            # If a symbolic attribute's value is missing, it is assigned the special value "?" and treated as a
            # legitimate symbolic value
            dataset[attr].fillna('?')
            # Decoding
            dataset[attr] = dataset[attr].str.decode('utf-8')

    return dataset

# Verify if an attribute is categorical
def is_categorical(array_like):
    return array_like.dtype.name == 'category' or array_like.dtype.name == 'object'

# Verify if an attribute is numerical
def is_numerical(array_like):
    return array_like.dtype.name == 'int64' or array_like.dtype.name == 'float64'

# Utility function to request user input
def make_selection(title, choices, prompt='Select one of the choices above'):
    print(title)
    print('-'*len(title))
    for choice in choices:
        print(str(choice) + ' - ' + choices[choice])

    selection_valid = False

    while (not selection_valid):
        selection = input(prompt  + ': ')

        if(selection in choices):
            selection_valid = True
        else:
            print('Error: Unrecognized option. Try again.')

    return selection

# RULE INTERPRETER
# Methods to interpret the rules: exporting them to a .txt file and importing them from a .txt file
def export_rules(RS, filename):
    with open(filename, 'w') as my_file:
        for R in RS:
            rule_text = "RULE " + str(R['id']) + ": IF '"
            countersize = 0
            for cond in R['conditions']:
                if cond[-1] == 'True':
                    rule_text = rule_text + str(cond[0]) + "' IS '" + str(cond[-1]) + "'"
                elif cond[-1] == 'symbolic':
                    rule_text = rule_text + str(cond[0]) + "' IS '" + str(cond[1]) + "'"
                elif cond[-1] == 'numeric':
                    rule_text = rule_text + str(cond[0]) + "' IN RANGE " + "[" + str(cond[1]) + ", " + str(
                        cond[2]) + "]"

                if countersize < (len(R['conditions']) - 1):
                    rule_text = rule_text + " AND '"
                countersize = countersize + 1

            rule_text = rule_text + " THEN CLASS: '" + str(R['class'] + "'")
            rule_text = rule_text + " WITH PRECISION OF " + str(R['precision']) + " AND COVERAGE " + str(R['covered'])
            my_file.write(rule_text + "\n\n")
        my_file.close()

def import_rules(filename):
    recreated_RS = []
    if os.path.exists(filename):
        with open(filename) as my_file:
            text = my_file.read()
            my_file.close()

        results = text.split("\n\n")
        counter = 0
        for rule in results[:-1]:
            rule_id = rule[5:rule.index(":")]
            str_rule_class = "THEN CLASS: '"
            rule_class = find_between(rule, str_rule_class, "' WITH PRECISION OF")

            rule_conditions = []
            conditions = rule.split(" AND '")
            conditions[-1] = conditions[-1][:(conditions[-1].index(str_rule_class)-1)]
            for c in conditions:
                if 'IF' in c:
                    c = c[rule.index(":")+6:]
                if 'RANGE' in c:
                    c_name = c[:c.index("' IN")]
                    c_lower = float(find_between(c, '[', ','))
                    c_upper = float(find_between(c, ',', ']'))
                    if c_lower == 'nan' or c_lower == 'NaN':
                        c_lower = np.nan
                    elif c_upper == 'nan' or c_upper == 'NaN':
                        c_upper = np.nan
                    rule_conditions.append((c_name, c_lower, c_upper, 'numeric'))
                elif 'True' in c:
                    c_name = c[:c.index("' IS")]
                    rule_conditions.append((c_name, 'True'))
                else:
                    c_name = c[:c.index("' IS")]
                    str_symbolic_value = "IS '"
                    c_value = c[c.index(str_symbolic_value) + len(str_symbolic_value):-1]
                    rule_conditions.append((c_name, c_value, 'symbolic'))
            dictionary = {'id': rule_id, 'class': rule_class, 'conditions': rule_conditions}
            recreated_RS.append(dictionary)
            counter = counter + 1
    else:
        print("The file with the rules was not found\n")

    return recreated_RS

def find_between(s, first, last):
    try:
        start = s.index(first) + len(first)
        end = s.index(last, start)
        return s[start:end]
    except ValueError:
        return ""



