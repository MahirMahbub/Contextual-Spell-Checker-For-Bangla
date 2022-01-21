import importlib
import os
import traceback
from typing import List, Optional, Any


class Interface(type):

    def __init__(self, name, bases, namespace):
        for base in bases:
            must_implement = getattr(base, 'abstract_methods', [])
            class_methods = getattr(self, 'all_methods', [])
            for method in must_implement:
                if method not in class_methods:
                    err_str = """Can't create abstract class {name}!
                    {name} must implement abstract method {method} of class {base_class}!""". \
                        format(name=name,
                               method=method,
                               base_class=base.__name__)
                    raise TypeError(err_str)

    def __new__(metaclass, name, bases, namespace):
        namespace['abstract_methods'] = Interface._get_abstract_methods(namespace)
        namespace['all_methods'] = Interface._get_all_methods(namespace)
        cls = super().__new__(metaclass, name, bases, namespace)
        return cls

    def _get_abstract_methods(namespace):
        return [name for name, val in namespace.items() if callable(val) and getattr(val, '__isabstract__', False)]

    def _get_all_methods(namespace):
        return [name for name, val in namespace.items() if callable(val)]


class Singleton:
    """
    A non-thread-safe helper class to ease implementing singletons.
    This should be used as a decorator -- not a metaclass -- to the
    class that should be a singleton.

    The decorated class can define one `__init__` function that
    takes only the `self` argument. Also, the decorated class cannot be
    inherited from. Other than that, there are no restrictions that apply
    to the decorated class.

    To get the singleton instance, use the `instance` method. Trying
    to use `__call__` will result in a `TypeError` being raised.

    """

    def __init__(self, decorated):
        self._decorated = decorated

    def instance(self):
        """
        Returns the singleton instance. Upon its first call, it creates a
        new instance of the decorated class and calls its `__init__` method.
        On all subsequent calls, the already created instance is returned.

        """
        try:
            return self._instance
        except AttributeError:
            self._instance = self._decorated()
            return self._instance

    def __call__(self):
        # raise TypeError('Singletons must be accessed through `instance()`.')
        return self.instance()

    def __instancecheck__(self, inst):
        return isinstance(inst, self._decorated)


def abstract(func):
    func.__isabstract__ = True
    return func


def create_instance(class_str: str, arguments: Optional[List[str]] = "") -> Any:
    """
    Create a class instance from a full path to a class constructor
    :param arguments: Optional List of arguments as string for the class's __init__() method
    :param class_str: module name plus '.' plus class name. For example, "a.b.ClassB.ClassB('World')"
    :return: an instance of the class specified.
    """

    class_str += "(" + ",".join(arguments) + ")"
    # print(class_str)
    try:
        if "(" in class_str:
            full_class_name, args = class_name = class_str.rsplit('(', 1)
            args = '(' + args
        else:
            full_class_name = class_str
            args = ()
        module_path, _, class_name = full_class_name.rpartition('.')
        mod = importlib.import_module(module_path)
        klazz = getattr(mod, class_name)
        alias = class_name + "Alias"
        instance = eval(alias + args, {alias: klazz})
        return instance
    except (ImportError, AttributeError) as e:
        raise ImportError(class_str)


def generate_json() -> None:
    import json
    dict = {
        "MLM": {
            "model": "bangla-bert-base",
            "controller": "BanglaBertMaskedModelController"
        },
        "NER": {
            "model": "mbert-bengali-ner",
            "controller": "BanglaBertNERModelController"
        }
    }
    with open('config.json', 'w') as f:
        json.dump(dict, f)


def exit_on_temp_fail() -> None:
    traceback.print_exc()
    os._exit(os.EX_TEMPFAIL)


def exit_on_data_err() -> None:
    traceback.print_exc()
    os._exit(os.EX_DATAERR)
