class Fan:
    power = lambda: 0

class Mk11(Fan):
    power = lambda flow: flow / 300 * 3.6 * 12

class orion1238(Fan):
    power = lambda flow: 21.4 * 2

#https://www.digikey.com/en/products/detail/delta-electronics/QFR1212GHEEVT/11201149
class qfr1212(Fan):
    power = lambda flow: 21.6 * 2
