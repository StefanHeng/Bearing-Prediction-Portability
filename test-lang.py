from collections import OrderedDict
import json

from icecream import ic

if __name__ == '__main__':
    # ic(json.dumps(OrderedDict(b=2, a=1)))

    x = {1: 2, 3: 4, 4: 3, 2: 1, 0: 0}
    ic({k: v for k, v in sorted(x.items(), key=lambda item: item[1])})

