% ===
% Produces CM200-CZ efficiency
% Data from: https://docs.google.com/presentation/d/1slilysASGEY36InsfS_VsN1nQvTZb4M8DncyWaOJRAM/edit?slide=id.g36c722c76c4_0_28#slide=id.g36c722c76c4_0_28
% Mk 10 to 11 KT slides. 
% NOTE: This is probably from Cascadia, cannot find on a datasheet.
% ===

function loss=inverter_loss (V,I, PWMfreq)
    arguments
        V double
        I double
        PWMfreq uint32 = 12
    end

    switch PWMfreq
    case 8
        a = 0.011046;
        b = 0.485276;
        c = 1.568988;
        offset = 213.8;
    case 10
        a = 0.005703;
        b = 0.565076;
        c = 1.617233;
        offset = 262.7;
    otherwise % 12kHz
        a = 0.002835;
        b = 0.665888;
        c = 1.651389;
        offset = 271.7;
    end

    loss = (a.*(V.^ b).*(I.^ c)) + offset;
end