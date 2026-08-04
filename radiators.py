import loop_helper as helper

# --- General Geometry Constants [m] ---
M_PER_TUBE = 0.0085  # 8.5 mm tube pitch
M_PER_FIN = 0.0008   # 0.8 mm fin/louver pitch


class Radiator:
    """
    Base class that calculates static mathematical attributes for all of 
    its subclasses automatically upon their creation.
    """
    # Fallback base attributes
    L = 0.0
    W = 0.0
    D = 0.0
    flow = lambda lpm: lpm * lpm * 0.07 * 0.07 
    
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        
        # Intermediate Calculations
        finN = cls.L / M_PER_FIN
        cls.tubeB = 0.0013       # shorter side of tube cross section (1.3 mm)
        finH = M_PER_TUBE - cls.tubeB
        tubeL = cls.L            # length of tube
        
        # Publicly accessible static properties requested
        cls.tubeA = cls.D       # longer side of tube cross section
        cls.tubeN = (cls.W - 0.01) / M_PER_TUBE
        
        cls.Awater = 2.0 * (cls.tubeA + cls.tubeB) * tubeL * cls.tubeN
        cls.Across = cls.L * (cls.W - cls.tubeN * cls.tubeB)
        cls.Aair = 2.0 * cls.tubeN * cls.L * cls.D + 2.0 * cls.tubeN * finN * finH * cls.D


class blackice_280(Radiator):
    # Radiator size
    L = 0.24
    W = 0.12
    D = 0.054


class monsta_280(Radiator):
    # Radiator size
    L = 0.24
    W = 0.12
    D = 0.084


class KX450F_left(Radiator):
    # Radiator size
    L = helper.in_to_m(11)
    W = helper.in_to_m(5)
    D = helper.in_to_m(1)


radiator_list = [blackice_280, monsta_280]
