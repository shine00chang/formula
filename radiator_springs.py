def blackice_280():
    # Radiator size
    L = 0.24
    W = 0.12
    D = 0.046
    finN = 520      # number of fins between two tubes
    finH = 0.147 * 0.0254 # gap between a fin and the one above it
    
    # NOTE: radiator splits flow into many parallel tubes, before rejoining. 
    tubeA = D       # longer side of tube cross section
    tubeB = 0.0013  # shorter side of tube cross section
    tubeL = 0.24    # length of tube
    tubeN = 14      # number of tubes
    
    # Area the water is convecting through
    Awater = 2.0 * (tubeA + tubeB) * tubeL * tubeN
        
    # Area the air is flowing through, required for air velocity
    Across = L * (W - tubeN * tubeB)
    
    # Area the air is convecting through
    Aair = 2.0 * tubeN * L * D + 2.0 * tubeN * finN * finH * D
    
    return Awater, Aair, Across, tubeA, tubeB, tubeN, D
