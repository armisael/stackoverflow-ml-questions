import sys
import numpy as np
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.cross_validation import KFold
from sklearn.tree import DecisionTreeClassifier

from settings import *
from helpers import is_number
from functions import get_user_by_name, get_interesting_user, \
    build_data_set, vectorize, StackOverflow


print "Running with feature level", FEATURES_LEVEL

user = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_USER
if not is_number(user):
    elem = get_user_by_name(user) if user else get_interesting_user()
    if elem is None:
        print "User '%s' not found" % user
        exit(1)
    user_id = int(elem.get("Id"))
else:
    user_id = int(user)
print "Building ML model for user %d" % user_id


instances, classes, tags_cnt = build_data_set(user_id, FEATURES_LEVEL == 0)

print "Set built, %d(%d+%d) (ratio: %.2f)" % (len(classes),
                                              classes.count(Classes.INTERESTED),
                                              classes.count(Classes.UNKNOWN),
                                              ML_DATA_RATIO)

if FEATURES_LEVEL == 0:
    stackoverflow_features = []
    tot_features = sum([tags_cnt[k] for k in tags_cnt])
    for k in tags_cnt:
        if 1.*tags_cnt[k]/tot_features >= 0.002:
            stackoverflow_features.append(k)
    print "StackOverflow features:", stackoverflow_features


mapping, X = vectorize(instances)

if FEATURES_LEVEL == 0:
    stackoverflow_tags = np.zeros(len(mapping))
    for k in stackoverflow_features:
        stackoverflow_tags[mapping[k]] = 1.

y = np.array(classes)
kf = KFold(len(classes), k=4)

if FEATURES_LEVEL > 0:
    classifiers = {
        'knn-15':   KNeighborsClassifier(15, weights='distance'),
        'svc':  SVC(C=1.0, coef0=0.0, degree=3, gamma=0.5, kernel='rbf', probability=False,
                     shrinking=True, tol=0.001),
        'tree': DecisionTreeClassifier(max_depth=10),
    }
else:
    classifiers = {
        'stackoverflow.com': StackOverflow(stackoverflow_tags)
    }

for clf_name in classifiers:
    clf = classifiers[clf_name]
    print "\n" * 2
    print "Classifying with", clf_name
    for train_index, test_index in kf:
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        clf.fit(X_train, y_train)
        z = clf.predict(X_test)

        cm = confusion_matrix(np.array(y_test), z)
        print cm
        correct = cm[0][0] + cm[1][1]
        wrong = cm[1][0] + cm[0][1]
        print 100. * correct / (wrong+correct), "%"


#Searching interesting user          100% ||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||| 100000
#756695
#Building ML model for user 22656
#Building data set                   100% ||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||| 100000
#6479788
#Set built, 39725(15890+23835) (ratio: 1.50)
#Building feature mapping            100% |||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||| 39725
#Building BOOL vectors               100% ||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||| 8849
#
#
#
#
#Classifying with stackoverflow.com
#[[3517  456]
# [4345 1613]]
#51.6564293626 %
#[[3698  274]
# [4710 1249]]
#49.813714631 %
#[[3754  218]
# [4862 1097]]
#48.8470446078 %
#[[3844  129]
# [4852 1107]]
#49.8489730165 %
#
#Classifying with tree
#[[3159  814]
# [1595 4363]]
#75.7426241063 %
#[[3433  539]
# [1504 4455]]
#79.4280535696 %
#[[3561  411]
# [1369 4590]]
#82.0763266539 %
#[[3617  356]
# [1235 4724]]
#83.9810712847 %







#StackOverflow features: ['wcf', 'arrays', 'osx', 'vb.net', 'ruby-on-rails', 'algorithm', 'asp.net', 'visual-studio', 'regex', 'c++', 'actionscript-3', 'jquery', 'cocoa', 'ajax', 'performance', 'sql-server', 'facebook', 'web-development', 'visual-studio-2010', 'xml', 'wpf', 'image', 'linq', 'python', 'c#', 'flash', 'string', 'json', 'web-services', 'multithreading', 'django', 'xcode', 'javascript', 'android', 'cocoa-touch', 'ruby', 'ruby-on-rails-3', 'sql', 'iphone', 'html', 'silverlight', 'java', 'flex', '.net', 'asp.net-mvc', 'mysql', 'objective-c', 'eclipse', 'php', 'linux', 'database', 'perl', 'css', 'ios', 'winforms', 'windows', 'c']

#cocoa-touch 14268 0.00241194409321
#web-development 14390 0.00243256766899
#performance 15275 0.00258217311632
#wcf 15464 0.00261412275423
#string 15560 0.00263035114173
#eclipse 15791 0.00266940069917
#silverlight 15916 0.00269053141207
#xcode 15925 0.0026920528234
#multithreading 16388 0.00277032098398
#visual-studio 16439 0.00277894231484
#arrays 17312 0.00292651921373
#winforms 17843 0.00301628248213
#linux 19693 0.00332901703305
#vb.net 20510 0.00346712737256
#ios 20944 0.00354049320774
#django 22232 0.00375822407346
#windows 23623 0.00399336664661
#database 23814 0.00402565437592
#regex 23915 0.00404272799194
#ajax 24527 0.0041461839623
#asp.net-mvc 26173 0.00442443318976
#xml 27893 0.00471519179925
#ruby 29497 0.00498634110718
#wpf 32958 0.00557140828594
#sql-server 33214 0.00561468398596
#c 39098 0.00660934890357
#css 44339 0.00749531743402
#ruby-on-rails 48298 0.00816456937298
#objective-c 52160 0.00881742387872
#sql 54625 0.0092341215371
#html 57841 0.00977777251858
#mysql 60709 0.0102625955953
#python 71673 0.0121160126852
#c++ 84777 0.0143311875799
#asp.net 88938 0.0150345867509
#.net 89646 0.0151542711087
#android 93247 0.0157630046859
#iphone 96752 0.0163555098756
#jquery 109129 0.0184477885441
#javascript 126297 0.0213499651766
#php 142125 0.0240256205668
#java 153561 0.0259588272285
#c# 211338 0.0357257808221
