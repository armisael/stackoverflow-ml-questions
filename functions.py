from lxml import etree
from copy import copy

import numpy as np
import nltk

from settings import *
from helpers import fast_iter, split_tags, build_pbar, Cache


def get_user_by_name(name):
    """ Given the user name, returns its XML entry (parsing users.xml) """

    context = etree.iterparse(DATASET_USERS, events=('end',), tag=("row"))
    
    def _matches(elem):
        if elem.get("DisplayName") == name:
            raise StopIteration("Found", elem)

    try:
        fast_iter(context, _matches)
    except StopIteration, e:
        return e.args[1]
    return None


def get_interesting_user():
    """ Returns the user with bigger reputation (excluded user with id=1) """

    pbar = build_pbar('Searching interesting user', TOT_USERS)
    context = etree.iterparse(DATASET_USERS, events=('end',), tag=("row"))
    _vars = {'max_rep': -1, 'user': None, 'cnt': 0}

    def _elab(elem, _vars):
        pbar.update(_vars['cnt'])
        _vars['cnt'] += 1

        if int(elem.get("Id")) == 1: return
        rep = int(elem.get("Reputation"))
        if _vars['max_rep'] < rep:
            _vars['max_rep'] = rep
            if _vars['user'] is not None: del _vars['user']
            _vars['user'] = copy(elem)

    fast_iter(context, _elab, _vars)
    pbar.finish()
    return _vars['user']


def build_data_set(user_id, count_tags=False):
    """ Given a user id, returns a tuple composed by the classifier instances
    and their classes. """
    pbar = build_pbar('Building data set', TOT_POSTS)
    context = etree.iterparse(DATASET_POSTS, events=('end',), tag=('row'))
    _vars = {'instances': [], 'classes': [], 'cache': Cache(), 'cnt': 0,
             'cnt_instances': {Classes.INTERESTED: 0, Classes.UNKNOWN: 0},
             'tags_cnt': {}}

    def _elab(elem, _vars):
        if _vars['cnt'] % PROGRESS_UPDATE == 0:
            pbar.update(_vars['cnt'])
        _vars['cnt'] += 1

        owner_sid = elem.get("OwnerUserId")
        post_id = int(elem.get("Id"))
        _vars['cache'].add('tags', post_id, split_tags(elem.get("Tags")))
        if count_tags:
            for t in _vars['cache'].get('tags', post_id):
                if t not in _vars['tags_cnt']:
                    _vars['tags_cnt'][t] = 0
                _vars['tags_cnt'][t] += 1

        if owner_sid is None: return

        post_class = extract_class(elem, user_id)
        if post_class == Classes.INTERESTED:
            _vars['instances'].append(extract_features(elem, _vars['cache']))
            _vars['classes'].append(post_class)
            _vars['cnt_instances'][Classes.INTERESTED] += 1
        elif _vars['cnt_instances'][Classes.UNKNOWN] < _vars['cnt_instances'][Classes.INTERESTED] * ML_DATA_RATIO:
            _vars['instances'].append(extract_features(elem, _vars['cache']))
            _vars['classes'].append(post_class)
            _vars['cnt_instances'][Classes.UNKNOWN] += 1


    fast_iter(context, _elab, _vars)
    pbar.finish()
    return _vars['instances'], _vars['classes'], _vars['tags_cnt']


def extract_features(elem, cache):
    """ Given an XML post, returns a list of features """
    post_id = int(elem.get("Id"))
    parent_sid = elem.get("ParentId")
    cache_id = post_id if parent_sid is None else int(parent_sid)
    features = []

    tags = cache.get('tags', cache_id)
    if tags is None:
        tags = []
        cache.register('tags', cache_id, lambda x, y: y.extend(x), features)
    features.extend(tags)

    if FEATURES_LEVEL >= 2:
        text = elem.get("Title", "")
        if FEATURES_LEVEL >= 3:
            text += " " + elem.get("Body", "")
        t_text = nltk.word_tokenize(text)
        t_text_tagged = nltk.pos_tag(t_text)

        keywords = []
        for word, pos in t_text_tagged:
            if pos == "NN" and \
               any([word[x].upper() == word[x] for x in range(min(3, len(word)))]) and \
               word.lower() not in NOT_NAMES:
                keywords.append(word.lower())
        features.extend(keywords)

    return features


def extract_class(elem, user_id):
    """ Given an XML post, returns its class. Ideal:
    EXPERT: answered questions;
    NEWBIE: asked questions;
    INTERESTED: unanswered questions with preferred tags;
    HATES_IT: unanswered questions with dislike tags;
    UNKNOWN: everything else.
    unfortunately we don't have any information about preferred user's tags.
    We can't therefore classify unanswered questions, which means that
    most of the classes are meaningless, so we have to use such classification:
    """

    owner_sid = elem.get("OwnerUserId")
    if owner_sid is None: return Classes.UNKNOWN
    if int(owner_sid) == user_id:
        return Classes.INTERESTED
    return Classes.UNKNOWN


def vectorize(*vectors):
    total = sum([len(v) for v in vectors])
    mapping = {}
    cnt = 0
    pbar = build_pbar("Building feature mapping", total)
    for v in vectors:
        for instance in v:
            pbar.update(pbar.currval+1)
            for e in instance:
                if e not in mapping:
                    mapping[e] = cnt
                    cnt += 1
    pbar.finish()

    result = [mapping]
    for v in vectors:
        result.append(np.zeros((len(v), cnt)))

    pbar = build_pbar("Building BOOL vectors", total)
    for i, v in enumerate(vectors):
        for j, instance in enumerate(v):
            pbar.update(pbar.currval+1)
            for k in instance:
                result[i+1][j][mapping[k]] = 1.
    pbar.finish()

    return tuple(result)


class StackOverflow(object):
    def __init__(self, tags):
        self.tags = tags

    def fit(self, X, y):
        pass

    def predict(self, X):
        y = np.zeros(len(X)) + 4.
        for i, x in enumerate(X):
            if any(x * self.tags):
                y[i] = 2.
        return y
