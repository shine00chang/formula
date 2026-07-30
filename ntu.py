import numpy as np

# physical constants
def rhowater(): return 1000.0
def rhoair(): return 1.225
def cwater(): return 4184.0
def cair(): return 1000.0
def viscositywater(): return 0.7 * 1e-6
def viscosityair(): return 1.6 * 1e-5
def kwater(): return 0.6
def kair(): return 0.026


# See: https://www.lamar.edu/engineering/_files/documents/mechanical/dr.-fan-publications/2012/No%2031%202012%20ASEE%20Radiator%20Final.pdf
def ntu_method(radiator, Qwater, Qair):
    Awater, Aair, Across, tubeA, tubeB, tubeN, D = radiator()
    Depth = D
    
    Ch = Qwater * rhowater() * cwater()
    Cc = Qair * rhoair() * cair()
    C = min(Ch, Cc)
    Cr = min(Cc, Ch) / max(Cc, Ch)
    
    # hwater
    Dhydraulic = 4.0 * tubeA * tubeB / (2.0 * tubeA + 2.0 * tubeB)
    vwater = Qwater / (tubeA * tubeB * tubeN)
    Rewater = vwater * Dhydraulic / viscositywater()
    Nuwater = 5.6 # refer to table 8.1 in 105D
    hwater = Nuwater * kwater() / Dhydraulic
    
    # hair
    vair = Qair / Across
    Reair = vair * Depth / viscosityair()
    Prair = 0.7
    Nuair = 0.664 * (Reair**0.5) * (Prair**0.33)
    hair = Nuair * kair() / Depth
    uair = 0.99 # TODO: though from Luke's code, this radiator gives about 0.99 efficiency.
    
    # overall heat transfer coefficient
    UA = 1.0 / ( 1.0/(uair * hair * Aair) + 1.0/(hwater * Awater) )
    NTU = UA / C
    
    # Cross flow, both fluids unmixed
    e = 1.0 - np.exp( (np.exp(-(NTU**0.78) * Cr) - 1) / (Cr * (NTU**-0.22)) )
    return e, NTU, hwater, hair, Nuair, Reair, C


