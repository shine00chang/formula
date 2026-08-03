# --- General Geometry Constants [m] ---
M_PER_TUBE = 0.0085  # 8.5 mm tube pitch
M_PER_FIN = 0.0008   # 0.8 mm fin/louver pitch


def blackice_280():
    # Radiator size
    L = 0.24
    W = 0.12
    D = 0.054
    finN = L / M_PER_FIN
    tubeB = 0.0013       # shorter side of tube cross section (1.3 mm)
    finH = M_PER_TUBE - tubeB
    
    tubeA = D            # longer side of tube cross section
    tubeL = L            # length of tube
    tubeN = (W - 0.01) / M_PER_TUBE

    Awater = 2.0 * (tubeA + tubeB) * tubeL * tubeN
    Across = L * (W - tubeN * tubeB)
    Aair = 2.0 * tubeN * L * D + 2.0 * tubeN * finN * finH * D
    
    return Awater, Aair, Across, tubeA, tubeB, tubeN, D

def monsta_280():
    # Radiator size
    L = 0.24
    W = 0.12
    D = 0.084
    finN = L / M_PER_FIN
    tubeB = 0.0013       # shorter side of tube cross section (1.3 mm)
    finH = M_PER_TUBE - tubeB
    
    tubeA = D            # longer side of tube cross section
    tubeL = L            # length of tube
    tubeN = (W - 0.01) / M_PER_TUBE

    Awater = 2.0 * (tubeA + tubeB) * tubeL * tubeN
    Across = L * (W - tubeN * tubeB)
    Aair = 2.0 * tubeN * L * D + 2.0 * tubeN * finN * finH * D
    
    return Awater, Aair, Across, tubeA, tubeB, tubeN, D


