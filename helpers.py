from settings import *

from progressbar import Percentage, ProgressBar, Bar, AnimatedMarker, Counter


def fast_iter(context, func, *args, **kwargs):
    for event, elem in context:
        func(elem, *args, **kwargs)
        elem.clear()
        while elem.getprevious() is not None:
            del elem.getparent()[0]
    del context


def is_number(s):
    if s is None: return False
    try:
        float(s)
        return True
    except ValueError:
        return False


def split_tags(tags):
    if tags is None: return []
    return tags[1:-1].split("><")


def build_pbar(text, max_val):
    text += " " * (PROGRESS_PADDING - len(text))
    widgets = [text, ' ', Percentage(), ' ',
               Bar(marker=AnimatedMarker()), ' ', Counter()]
    return ProgressBar(widgets=widgets, maxval=max_val).start()


class Cache(object):
    def __init__(self):
        self.data = {}
        self.registrations = {}

    def add(self, type, key, value):
        if type not in self.data:
            self.data[type] = {}
        self.data[type][key] = value
        self._notify(type, key, value)

    def get(self, type, key):
        if type not in self.data:
            return None
        if key not in self.data[type]:
            return None
        return self.data[type][key]

    def register(self, type, key, method, *args, **kwargs):
        if type not in self.registrations:
            self.registrations[type] = {}
        self.registrations[type][key] = (method, args, kwargs)

    def _notify(self, type, key, value):
        if type not in self.registrations:
            return
        if key not in self.registrations[type]:
            return
        method, args, kwargs = self.registrations[type][key]
        method(value, *args, **kwargs)
