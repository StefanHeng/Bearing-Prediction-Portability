from collections import OrderedDict
import json

from icecream import ic

if __name__ == '__main__':
    ic(json.dumps(OrderedDict(b=2, a=1)))
