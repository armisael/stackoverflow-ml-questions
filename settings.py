TESTING = False
FEATURES_LEVEL = 0

if TESTING:
    DATASET_DIR = '/Users/stefanop/Desktop/092011 Meta Stack Overflow/'
    DEFAULT_USER = 811
    TOT_POSTS = 85123
    TOT_USERS = 31535
else:
    DATASET_DIR = '/Users/stefanop/Desktop/092011 Stack Overflow/'
    DEFAULT_USER = 22656
    TOT_POSTS = 6479788
    TOT_USERS = 756695

DATASET_USERS = DATASET_DIR + 'users.xml'
DATASET_POSTS = DATASET_DIR + 'posts.xml'
DATASET_COMMENTS = DATASET_DIR + 'comments.xml'

ML_DATA_RATIO = 1.5
ML_TRAINING_RATIO = 0.7

PROGRESS_UPDATE = 1000
PROGRESS_PADDING = 35

NOT_NAMES = ['i', 'you', 'he', 'she', 'they', 'we', '%', '='
             '"', ')', '(', '[', ']', '{', '}', 'please', 'today']

class Classes:
    EXPERT = 0
    NEWBIE = 1
    INTERESTED = 2
    HATES_IT = 3
    UNKNOWN = 4
