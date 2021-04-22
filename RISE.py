from utils import *
from math import pow
import random
import math
import time

# RISE Algorithm
class RISE:

    def __init__(self):
        self.dataset = 0
        self.data_test = 0
        self.RS = []
        self.wins = []
        self.misclassified = []
        self.RULE_FILENAME = ""
        self.attributes = ""
        self.class_column = ""
        self.classes = []

    # This method creates the initial rule set based on the training instances
    def initialize(self, data_x, data_y):
        data_x.reset_index(drop=True, inplace=True)
        data_y.reset_index(drop=True, inplace=True)
        self.dataset = pd.DataFrame(pd.concat([data_x, data_y], axis=1))
        # Preprocessing is applied
        self.dataset = preprocessing(self.dataset)
        # Some relevant variables are obtained
        self.attributes = self.dataset.drop(self.dataset.columns[-1], axis=1).columns
        self.class_column = self.dataset.columns[-1]
        self.classes = self.dataset[self.class_column].unique()

        id = 0
        # For each training instance, model its rule
        for i in range(self.dataset.shape[0]):
            irule = []
            # For each attribute of the instance, check if it's symbolic or numeric and add it to the rule conditions
            for attr in self.attributes:
                if is_categorical(self.dataset[attr]):
                    irule.append((attr, self.dataset.iloc[i][attr], 'symbolic'))
                elif is_numerical(self.dataset[attr]):
                    irule.append((attr, self.dataset.iloc[i][attr], self.dataset.iloc[i][attr], 'numeric'))
            dictionary = {'id': id, 'class': self.dataset.iloc[i][self.class_column], 'conditions': irule, 'covered': [i]}
            self.RS.append(dictionary)
            id = id + 1

        self.print_rules()

    # Main algorithm flow
    def fit(self, data_x, data_y):
        self.initialize(data_x, data_y)

        # Compute the total accuracy
        oldacc = self.compute_init_acc()
        print("Initial accuracy is " + str(oldacc))

        while True:
            start = time.time()
            print("Starting round...")
            # round_improvement and stop variablees will control the while loop
            round_improvement = False
            stop = True

            # For each rule R in RS
            for R in self.RS:
                # Find the nearest example E of R's class to R not already covered by it
                found, mindist, index_E = self.find_nearest_not_covered(R)
                # If not found, R is omitted from this round
                if not found:
                    continue

                # Find most specific generalization
                generalized, nR = self.most_specific_generalization(R, self.dataset.iloc[index_E])
                # If the new generalized rule is the same as the original R, R is omitted from this round
                if not generalized:
                    continue

                # stop is set to false, meaning that a rule has been successfully generalized
                stop = False

                # RS' = RS with R replaced by R'
                nRS = self.RS.copy()
                # index_E is added to the list of instances covered by the new rule
                nR['covered'].append(index_E)
                nRS[self.RS.index(R)] = nR

                # Compute changes in accuracy by considering the new rule in the rule set
                newacc = self.compute_new_accuracy(nRS, nR)

                if newacc >= oldacc:
                    if newacc > oldacc:
                        # If the accuracy has been improved, round_improvement is set to True
                        print("Accuracy improved: " + str(newacc))
                        round_improvement = True

                    # Replace RS by new RS
                    self.RS = nRS
                    # If new rule is identical to another rule in RS then delete it from RS
                    is_identical, id_identical = self.identical_found(nR)
                    if is_identical:
                        # Also, compare the two lists of instances covered of both rules and, if they are different,
                        # append the non-shared instance indices into the list of the identical rule found
                        # rule
                        rule_identical = [r for r in self.RS if r['id'] == id_identical][0]
                        if rule_identical['covered'] != nR['covered']:
                            listofcovered = rule_identical['covered']
                            for i in nR['covered']:
                                if i not in rule_identical['covered']:
                                    listofcovered.append(i)
                            rule_identical['covered'] = listofcovered
                            self.RS[self.RS.index(rule_identical)] = rule_identical
                        # In case that the new rule that is going to be eliminated is the winner rule of some instance,
                        # change that winner rule to the identical found
                        self.change_wins(nR['id'], id_identical)
                        self.RS.pop(self.RS.index(nR))
                    oldacc = newacc
                    # If an accuracy of 100% is reached, break all the loops and terminate the training
                    if newacc == 1.0:
                        stop = True
                        break

            # The RS stops being examined if stop == True (meaning that no rule has been properly generalized in
            # a whole analysis round of the RS) or round_improvement == False (meaning that there has not been an
            # accuracy improvement in the whole analysis round of the RS)
            if stop == True or round_improvement == False:
                break
        print()
        end = time.time()
        print()
        print("Training time: " + str(round(end - start,2)) + "s")
        print("Final accuracy: " + str(oldacc))
        print("Final number of rules: " + str(len(self.RS)))
        print("In order to visualize the final rules, check " + str(self.RULE_FILENAME))
        self.reorder_ruleids()
        # Compute rule precisions
        self.compute_rules_precisions()
        # Export final RS into a .txt
        export_rules(self.RS, self.RULE_FILENAME)

    # Method to interpret the final RS from a .txt file, apply it to a test set and compute the test accuracy
    def test(self, test_x, test_y):
        test_acc = 0
        test_x.reset_index(drop=True, inplace=True)
        test_y.reset_index(drop=True, inplace=True)
        data_test = pd.concat([test_x, test_y], axis=1)
        # Preprocess test set
        data_test = preprocessing(data_test)
        self.data_test = data_test

        # Interpret rules from the .txt
        recreated_RS = import_rules(self.RULE_FILENAME)

        if recreated_RS:
            correctly_classified = 0
            # Find all the test instances covered by each rule (this is necessary for computing Laplace accuracies)
            recreated_RS = self.find_test_covered(recreated_RS)
            # For each test instance
            for i in range(data_test.shape[0]):
                # Find nearest rule (winner rule) to the example
                nearest_rule_id, mindist = self.find_nearest(recreated_RS, data_test.iloc[i])
                nearest_rule = [r for r in recreated_RS if r['id'] == nearest_rule_id][0]
                # Verify if the classification by the winner rule is accurate
                if nearest_rule['class'] == data_test.iloc[i][self.class_column]:
                    correctly_classified = correctly_classified + 1

            # Compute test accuracy
            test_acc = correctly_classified / data_test.shape[0]
        else:
            print("ERROR: There was a problem extracting the rules from the file.")

        return test_acc

    # Find all the test instances covered by each rule
    def find_test_covered(self, recreated_RS):
        for R in recreated_RS:
            conditions = R['conditions']
            covered = []
            for i in range(self.data_test.shape[0]):
                add = 0
                for c in conditions:
                    if c[-1] == 'numeric' and self.data_test.iloc[i][c[0]] >= c[1] \
                            and self.data_test.iloc[i][c[0]] <= c[2]:
                        add = add + 1
                    elif c[-1] == 'symbolic' and self.data_test.iloc[i][c[0]] == c[1]:
                        add = add + 1
                    elif c[-1] == 'True':
                        add = add + 1
                if add == len(conditions):
                    covered.append(i)
            R['covered'] = covered
            recreated_RS[recreated_RS.index(R)] = R
        return recreated_RS

    # Method to find the nearest instance of the same class not covered by a rule, which will be used
    # to generalize that rule
    def find_nearest_not_covered(self, rule):
        # Get the non-covered instances by R with the same class
        not_covered_same_class = []
        for i in range(self.dataset.shape[0]):
            if i not in rule['covered'] and rule['class'] == self.dataset.iloc[i][self.class_column]:
                not_covered_same_class.append(i)

        if not not_covered_same_class:
            return False, -1, -1

        # Compute distances between R and instances in not_covered_same_class in order to find the nearest instance
        # to the rule
        mindist = pow(10,10)
        nearest_index = -1
        for i in not_covered_same_class:
            dist = self.compute_distance(rule, self.dataset.iloc[i])
            if dist <= mindist:
                mindist = dist
                nearest_index = i

        return True, mindist, nearest_index

    # Method in which rule generalization is carried out
    def most_specific_generalization(self, rule, e):
        # A copy of the rule is created to generalize over it
        new_rule = rule.copy()
        new_cond = []

        for cond in rule['conditions']:
            attr = cond[0]
            type = cond[-1]
            if type == 'symbolic' and cond[1] != e[attr]:
                # A missing symbolic value in the instance or in the rule causes the corresponding condition to be
                # dropped. The second condition of the if statement covers the situation when either the rule or the
                # instance value is missing, no both of them
                new_cond.append((attr, 'True'))
            elif type == 'numeric' and e[attr] > cond[2]:
                    new_cond.append((attr, cond[1], e[attr], 'numeric'))
            elif type == 'numeric' and e[attr] < cond[1]:
                    new_cond.append((attr, e[attr], cond[2], 'numeric'))
            else:
                # Every other case is covered here, including when the attribute is set to True and
                # when the numeric attributes have missing values
                new_cond.append(cond)

        # If the generalization has the same conditions as the rule generalized, it's not considered for further
        # analysis. Else, it is returned
        if new_cond == rule['conditions']:
            return False, -1
        else:
            new_rule['conditions'] = new_cond
            return True, new_rule

    # Find nearest rule to an instance
    def find_nearest(self, nRS, e, mode='train'):
        mindist = pow(10,20)
        nearest_id = -1
        rules_dist = []
        rules_ids = []

        for R in nRS:
            dist = self.compute_distance(R, e, mode)
            rules_dist.append(dist)
            rules_ids.append(R['id'])
            if dist <= mindist:
                mindist = dist
                nearest_id = R['id']

        # Policy to follow when more than one rule are equally near to the instance
        if rules_dist.count(mindist) > 1:
            equally_near_ids = [i for i, d in zip(rules_ids, rules_dist) if d == mindist]
            nearest_id = self.equally_near_policy(equally_near_ids, nRS)

        return nearest_id, mindist

    # Compute initial accuracy
    def compute_init_acc(self):
        correctly_classified = 0
        for i in range(self.dataset.shape[0]):
            # Leave-one-out is used without verification, as at this point no rule covers more than one instance
            nRS = self.RS.copy()
            nRS.pop(i)

            # Find nearest rule to the example
            nearest_rule_id, mindist = self.find_nearest(nRS, self.dataset.iloc[i])
            # Set that rule as the winner ruler for the example
            self.wins.append((nearest_rule_id, mindist))
            nearest_rule = [r for r in nRS if r['id'] == nearest_rule_id][0]

            # Verify if the classification by the winner rule is accurate
            if nearest_rule['class'] == self.dataset.iloc[i][self.class_column]:
                correctly_classified = correctly_classified + 1
                self.misclassified.append(0)
            else:
                self.misclassified.append(1)
        totalacc = correctly_classified / self.dataset.shape[0]
        return totalacc

    # Compute new accuracy considering new R
    def compute_new_accuracy(self, nRS, R):
        correctly_classified, numerator_acc = 0, 0
        misclassified_local = self.misclassified.copy()
        wins_local = self.wins.copy()

        # The distance between the rule and each instance is computer. If lower than the distance of the current
        # winner rule of each instance, the winner rule is updated. The final winner rule will determine the increase
        # or decrease in accuracy
        for i in range(self.dataset.shape[0]):
            dist = self.compute_distance(R, self.dataset.iloc[i])

            #
            if dist < self.wins[i][1]:
                wins_local[i] = (R['id'], dist)
            # Policy to follow when more than one rule are equally near to the instance
            elif dist == self.wins[i][1] and R['id'] != self.wins[i][0]:
                r_ids = [R['id'], self.wins[i][0]]
                nearest_id = self.equally_near_policy(r_ids, nRS)
                wins_local[i] = (nearest_id, dist)

            index_r = int(wins_local[i][0])
            rule_wanted = [r for r in nRS if r['id'] == index_r][0]

            # If a previously misclassified example is now correctly classified, correctly_classified is incremented
            if self.misclassified[i] == 1 and rule_wanted['class'] == self.dataset.iloc[i][self.class_column]:
                correctly_classified = correctly_classified + 1
                misclassified_local[i] = 0
            # If a previously well-classified example is now misclassified, correctly_classified is decremented
            elif self.misclassified[i] == 0 and rule_wanted['class'] != self.dataset.iloc[i][self.class_column]:
                correctly_classified = correctly_classified - 1
                misclassified_local[i] = 1

            if rule_wanted['class'] == self.dataset.iloc[i][self.class_column]:
                numerator_acc = numerator_acc + 1

        # If the sum of increments and decrements is greater than or equal than 0, the new rule is adopted and
        # the relevant structures are updated
        if correctly_classified >= 0:
            self.wins = wins_local
            self.misclassified = misclassified_local
            totalacc = numerator_acc / self.dataset.shape[0]
            return totalacc
        else:
            return 0

    # Computing distance between a rule and an instance
    def compute_distance(self, r, e, mode='train'):
        totaldistance = 0
        for c in r['conditions']:
            # As attrdistance is set to 0, if the following conditions are not fulfilled several situtations are still
            # covered, for example when attrtype == True or when there's a missing numeric value
            attrdistance = 0
            attrtype = c[-1]
            attr = c[0]

            if attrtype == 'symbolic':
                attrdistance = self.compute_SVDM(c, e[attr])
            # For numeric attributes, distance is only computed if there's no missing values in the rule or the
            # instance. The distance from a missing numeric value to any other is 0
            elif attrtype == 'numeric' and not pd.isna(e[attr]) and \
                    not math.isnan(float(c[1])) and not math.isnan(float(c[2])):
                attrdistance = self.compute_num(c, e[attr], mode)

            totaldistance = totaldistance + pow(attrdistance,2)
        return totaldistance

    # Compute SVDM for categorical attributes based on bayesian probabilities
    def compute_SVDM(self, r, e):
        distallclasses = 0
        data = self.dataset

        for C in self.classes:
            class_instances = data[data[self.class_column] == C]

            # Bayes Theorem
            count_class = class_instances.shape[0]
            prob_class = class_instances.shape[0] / data.shape[0]

            countR_class = class_instances[class_instances[r[0]] == r[1]].shape[0]
            countR_total = data[data[r[0]] == r[1]].shape[0]
            probR_x_class = countR_class / count_class
            probR_x = countR_total / class_instances.shape[0]

            countE_class = class_instances[class_instances[r[0]] == e].shape[0]
            countE_total = data[data[r[0]] == e].shape[0]
            probE_x_class = countE_class / count_class
            probE_x = countE_total / class_instances.shape[0]

            if probR_x == 0:
                probR_class_x = 0
            elif probE_x == 0:
                probE_class_x = 0
            else:
                probR_class_x = probR_x_class * prob_class / probR_x
                probE_class_x = probE_x_class * prob_class / probE_x

            opt = abs(probR_class_x - probE_class_x)
            distallclasses = distallclasses + pow(opt,2)

        return distallclasses

    # Compute distance for numerical attributes
    def compute_num(self, r, e, mode='train'):
        if mode == 'train':
            data = self.dataset
        elif mode == 'test':
            data = self.data_test

        rlower = r[1]
        rupper = r[2]
        emax = data[r[0]].max()
        emin = data[r[0]].min()

        dist = 0
        # condition e >= rlower and e <= rupper covered with dist = 0
        if e > rupper:
            dist = e - rupper / emax - emin
        elif e < rlower:
            dist = rlower - e / emax - emin

        return dist

    # Compute laplace accuracy
    def compute_laplace_acc(self, rule):
        positive_examples = 0
        negative_examples = 0
        laplace_acc = 0
        if not rule['covered']:
            return laplace_acc

        for i in rule['covered']:
            if self.dataset.iloc[i][self.class_column] == rule['class']:
                positive_examples = positive_examples + 1
            else:
                negative_examples = negative_examples + 1

        laplace_acc = (positive_examples + 1) / (positive_examples + negative_examples + self.classes.size)
        return laplace_acc

    # Implementation of the policy followed when more than one rule is the nearest to an instance
    def equally_near_policy(self, id_list, nRS):
        laplace_accs = []
        # Compute Laplace accuracies for all nearest rules
        for id in id_list:
            rule_wanted = [r for r in nRS if r['id'] == id][0]
            laplace_accs.append(self.compute_laplace_acc(rule_wanted))
        # If more than one rule has the same maximal Laplace accuracy, calcule the frequency of the class of each of
        # the rules in the dataset
        if laplace_accs.count(np.max(laplace_accs)) > 1:
            freq_classes = []
            for id in id_list:
                rule_wanted = [r for r in nRS if r['id'] == id][0]
                freq_classes.append(self.dataset[self.dataset[self.class_column] == rule_wanted['class']].shape[0])
                # If the tie remains, choose nearest rule at random
            if freq_classes.count(np.max(freq_classes)) > 1:
                nearest_id = random.choice(id_list)
            else:
                nearest_id = [i for i, x in zip(id_list, freq_classes) if x == np.max(freq_classes)][0]
        else:
            nearest_id = [i for i, x in zip(id_list, laplace_accs) if x == np.max(laplace_accs)][0]

        return nearest_id

    # Search for identical rule in the RS
    def identical_found(self, rule):
        RS_aux = self.RS.copy()
        RS_aux.pop(RS_aux.index(rule))
        for R in RS_aux:
            if self.dicts_equal(R,rule):
                return True, R['id']

        return False, -1

    # Check if two rules are identical
    def dicts_equal(self, rule1, rule2):
        ignored = ['id', 'covered']
        for k,v in rule1.items():
            if k not in ignored and rule2[k] != v:
                return False
        return True

    # Change winner rule if identical is found
    def change_wins(self, id_rule, id_identical):
        i = 0
        for w in self.wins:
            if w[0] == id_rule:
                waux = (id_identical, w[1])
                self.wins[i] = waux
            i = i + 1

    # Reorder rule ids at the end of the training phase
    def reorder_ruleids(self):
        for i in range(len(self.RS)):
            self.RS[i]['id'] = i

    # Compute rules precisions at the end of the training phase
    def compute_rules_precisions(self):
        for R in self.RS:
            correctly_classified = 0
            for c in R['covered']:
                if self.dataset.iloc[c][self.class_column] == R['class']:
                    correctly_classified = correctly_classified + 1
            rule_precision = correctly_classified / len(R['covered'])
            R['precision'] = rule_precision
        self.RS[self.RS.index(R)] = R

    # Print rules in the RS
    def print_rules(self):
        for R in self.RS:
            print("Rule " + str(R['id']) + ": " + str(R['class']) + ' -> ' + str(R['conditions'])
                  + " covers instance(s) with indices " + str(R['covered']))