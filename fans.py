class Fan:
    power = lambda: 0

class Mk11(Fan):
    power = lambda flow: flow / 300 * 3.6 * 12

class orion1238(Fan):
    power = lambda flow: 21.4 * 2
