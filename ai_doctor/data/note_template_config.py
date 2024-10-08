########################################################################################################################
# car
########################################################################################################################
# prices_dict = {'vhigh': 'very high', 'high': 'high', 'med': 'medium', 'low': 'low'}
# doors_dict = {'2': 'two', '3': 'three', '4': 'four', '5more': 'five or more'}
# persons_dict = {'2': 'two', '4': 'four', 'more': 'more than four'}
# lug_boot_dict = {'big': 'big', 'med': 'medium', 'small': 'small'}
# safety_dict = {'high': 'high', 'med': 'medium', 'low': 'low'}
# template_config_car = {
#     'pre': {
#         'doors': lambda x: doors_dict[x],
#         'persons': lambda x: persons_dict[x],
#         'lug_boot': lambda x: lug_boot_dict[x],
#         'safety_dict': lambda x: safety_dict[x],
#         'buying': lambda x: prices_dict[x],
#         'maint': lambda x: prices_dict[x]
#     }
# }
# template_car = 'The Buying price is ${buying}. ' \
#                'The Doors is ${doors}. ' \
#                'The Maintenance costs is ${maint}. ' \
#                'The Persons is ${persons}. ' \
#                'The Safety score is ${safety_dict}. ' \
#                'The Trunk size is ${lug_boot}.'
INT_MAX = 9999999999
INT_MIN = -9999999999
FLAG_NORM = "normal"  # "正常"
FLAG_ABNORM = "abnormal"  # "异常"
rule_yiduo = {
    "K1 F (D)": {"bins": [INT_MIN, 45, INT_MAX], "labels": [FLAG_NORM, FLAG_ABNORM]},
    "K2 F (D)": {"bins": [INT_MIN, 49, INT_MAX], "labels": [FLAG_NORM, FLAG_ABNORM]},
    "Km F (D)": {"bins": [INT_MIN, 46, INT_MAX], "labels": [FLAG_NORM, FLAG_ABNORM]},
    "K Max (Front)": {"bins": [INT_MIN, 48.2, INT_MAX], "labels": [FLAG_NORM, FLAG_ABNORM]},
    "K Max X (Front)": {"bins": [INT_MIN, 0.07, INT_MAX], "labels": [FLAG_NORM, FLAG_ABNORM]},
    "K Max Y (Front)": {"bins": [INT_MIN, -1, INT_MAX], "labels": [FLAG_ABNORM, FLAG_NORM]},
    "RMS (CF)": {"bins": [INT_MIN, 2, INT_MAX], "labels": [FLAG_NORM, FLAG_ABNORM]},
    "RMS HOA (CF)": {"bins": [INT_MIN, 0.4, INT_MAX], "labels": [FLAG_NORM, FLAG_ABNORM]},
    "RMS LOA (CF)": {"bins": [INT_MIN, 2, INT_MAX], "labels": [FLAG_NORM, FLAG_ABNORM]},

    "K1 B (D)": {"bins": [INT_MIN, -6.4, INT_MAX], "labels": [FLAG_ABNORM, FLAG_NORM]},
    "K2 B (D)": {"bins": [INT_MIN, -6.6, INT_MAX], "labels": [FLAG_ABNORM, FLAG_NORM]},
    "Km B (mm)": {"bins": [INT_MIN, -6.4, INT_MAX], "labels": [FLAG_ABNORM, FLAG_NORM]},

    "RMS (CB)": {"bins": [INT_MIN, 1, INT_MAX], "labels": [FLAG_NORM, FLAG_ABNORM]},
    "RMS HOA (CB)": {"bins": [INT_MIN, 0.25, INT_MAX], "labels": [FLAG_NORM, FLAG_ABNORM]},
    "RMS LOA (CB)": {"bins": [INT_MIN, 1.1, INT_MAX], "labels": [FLAG_NORM, FLAG_ABNORM]},

    "Pachy Apex:(CCT)": {"bins": [INT_MIN, 516, INT_MAX], "labels": [FLAG_ABNORM, FLAG_NORM]},
    "Pachy Min: (TCT)": {"bins": [INT_MIN, 511, INT_MAX], "labels": [FLAG_ABNORM, FLAG_NORM]},

    "Dist. Apex-Thin.Loc. [mm](Dist. C-T)": {"bins": [INT_MIN, 0.9, INT_MAX],
                                              "labels": [FLAG_NORM, FLAG_ABNORM]},
    "PachyMinX": {},
    "PachyMinY": {},
    "EccSph": {"bins": [INT_MIN, 0.6, INT_MAX], "labels": [FLAG_NORM, FLAG_ABNORM]},
    "RMS (Cornea)": {"bins": [INT_MIN, 2, INT_MAX], "labels": [FLAG_NORM, FLAG_ABNORM]},
    "RMS HOA (Cornea)": {"bins": [INT_MIN, 0.5, INT_MAX], "labels": [FLAG_NORM, FLAG_ABNORM]},
    "RMS LOA (Cornea)": {"bins": [INT_MIN, 2, INT_MAX], "labels": [FLAG_NORM, FLAG_ABNORM]},

    "C.Vol D 3mm": {},
    "C.Vol D 5mm": {},
    "C.Vol D 7mm": {},
    "C.Vol D 10mm": {},
    "BAD Df": {"bins": [INT_MIN, 2.6, INT_MAX], "labels": [FLAG_NORM, FLAG_ABNORM]},
    "BAD Db": {"bins": [INT_MIN, 2.6, INT_MAX], "labels": [FLAG_NORM, FLAG_ABNORM]},
    "BAD Dp": {"bins": [INT_MIN, 2.6, INT_MAX], "labels": [FLAG_NORM, FLAG_ABNORM]},
    "BAD Dt": {"bins": [INT_MIN, 2.6, INT_MAX], "labels": [FLAG_NORM, FLAG_ABNORM]},
    "BAD Da": {"bins": [INT_MIN, 2.6, INT_MAX], "labels": [FLAG_NORM, FLAG_ABNORM]},
    "BAD Dy": {"bins": [INT_MIN, 2.6, INT_MAX], "labels": [FLAG_NORM, FLAG_ABNORM]},
    "ISV": {"bins": [INT_MIN, 41, INT_MAX], "labels": [FLAG_NORM, FLAG_ABNORM]},
    "IVA": {"bins": [INT_MIN, 0.32, INT_MAX], "labels": [FLAG_NORM, FLAG_ABNORM]},
    "KI": {"bins": [INT_MIN, 1.07, INT_MAX], "labels": [FLAG_NORM, FLAG_ABNORM]},
    "CKI": {"bins": [INT_MIN, 1.03, INT_MAX], "labels": [FLAG_NORM, FLAG_ABNORM]},
    "IHA": {"bins": [INT_MIN, 21, INT_MAX], "labels": [FLAG_NORM, FLAG_ABNORM]},
    "IHD": {"bins": [INT_MIN, 0.016, INT_MAX], "labels": [FLAG_NORM, FLAG_ABNORM]},
    "R Min (mm)": {"bins": [INT_MIN, 7, INT_MAX], "labels": [FLAG_ABNORM, FLAG_NORM]},
    "Pachy Prog Index Min.": {"bins": [INT_MIN, 0.88, INT_MAX], "labels": [FLAG_NORM, FLAG_ABNORM]},
    "Pachy Prog Index Max.": {"bins": [INT_MIN, 1.58, INT_MAX], "labels": [FLAG_NORM, FLAG_ABNORM]},
    "Pachy Prog Index Avg.": {"bins": [INT_MIN, 1.08, INT_MAX], "labels": [FLAG_NORM, FLAG_ABNORM]},
    "ART Min.": {},
    "ART Max.": {"bins": [INT_MIN, 412, INT_MAX], "labels": [FLAG_ABNORM, FLAG_NORM]},
    "ART Avg.": {},
    "C.Volume(chamber volume)": {"bins": [INT_MIN, 210, INT_MAX], "labels": [FLAG_NORM, FLAG_ABNORM]},
    "Chamber angle": {"bins": [INT_MIN, 44, INT_MAX], "labels": [FLAG_NORM, FLAG_ABNORM]},
    "A.C.Depth Int": {"bins": [INT_MIN, 3.4, INT_MAX], "labels": [FLAG_NORM, FLAG_ABNORM]},
    "Elevation front": {"bins": [INT_MIN, 6, INT_MAX], "labels": [FLAG_NORM, FLAG_ABNORM]},
    "Elevation back": {"bins": [INT_MIN, 12.3, INT_MAX], "labels": [FLAG_NORM, FLAG_ABNORM]},
    "IOP [mmHg]": {},
    "DA [mm]": {},
    "A1T [ms]": {"bins": [INT_MIN, 7.15, INT_MAX], "labels": [FLAG_ABNORM, FLAG_NORM]},
    "A1V [m/s]": {"bins": [INT_MIN, 0.16, INT_MAX], "labels": [FLAG_NORM, FLAG_ABNORM]},
    "A2T [ms]": {"bins": [INT_MIN, 22.1, INT_MAX], "labels": [FLAG_NORM, FLAG_ABNORM]},
    "A2V [m/s]": {"bins": [INT_MIN, -0.3, INT_MAX], "labels": [FLAG_ABNORM, FLAG_NORM]},
    "HCT [ms]": {"bins": [INT_MIN, 17.25, INT_MAX], "labels": [FLAG_NORM, FLAG_ABNORM]},
    "PD [mm]": {"bins": [INT_MIN, 5.3, INT_MAX], "labels": [FLAG_NORM, FLAG_ABNORM]},
    "A1L [mm]": {},
    "HCDL [mm]": {},
    "A2L [mm]": {},
    "A1DeflAmp. [mm](A1DA)": {"bins": [INT_MIN, 0.095, INT_MAX], "labels": [FLAG_NORM, FLAG_ABNORM]},
    "HCDeflAmp. [mm](HCDA)": {"bins": [INT_MIN, 1, INT_MAX], "labels": [FLAG_NORM, FLAG_ABNORM]},
    "A2DeflAmp. [mm](A2DA)": {"bins": [INT_MIN, 0.11, INT_MAX], "labels": [FLAG_NORM, FLAG_ABNORM]},
    "A1DeflArea [mm^2]": {"bins": [INT_MIN, 0.19, INT_MAX], "labels": [FLAG_NORM, FLAG_ABNORM]},
    "HCDeflArea [mm^2]": {"bins": [INT_MIN, 4, INT_MAX], "labels": [FLAG_NORM, FLAG_ABNORM]},
    "A2DeflArea [mm^2]": {"bins": [INT_MIN, 0.25, INT_MAX], "labels": [FLAG_NORM, FLAG_ABNORM]},
    "A1ΔArcL [mm]": {},
    "HCΔArcL [mm]": {},
    "A2ΔArcL [mm]": {},
    "MaxIR [mm^-1]": {"bins": [INT_MIN, 0.18, INT_MAX], "labels": [FLAG_NORM, FLAG_ABNORM]},
    "DAR2": {"bins": [INT_MIN, 5, INT_MAX], "labels": [FLAG_NORM, FLAG_ABNORM]},
    "DAR1": {"bins": [INT_MIN, 1.55, INT_MAX], "labels": [FLAG_NORM, FLAG_ABNORM]},
    "ARTh": {"bins": [INT_MIN, 300, INT_MAX], "labels": [FLAG_ABNORM, FLAG_NORM]},
    "bIOP": {"bins": [INT_MIN, 14, INT_MAX], "labels": [FLAG_ABNORM, FLAG_NORM]},
    "IR [mm^-1]": {"bins": [INT_MIN, 9, INT_MAX], "labels": [FLAG_NORM, FLAG_ABNORM]},
    "SP A1": {"bins": [INT_MIN, 80, INT_MAX], "labels": [FLAG_ABNORM, FLAG_NORM]},
    "SSI": {"bins": [INT_MIN, 0.75, INT_MAX], "labels": [FLAG_ABNORM, FLAG_NORM]},
}

template_config_yiduo = {
    'preprocess': {
    }
}

template_yiduo_01 = ('性别是${性别}'
                     + 'BMI is ${BMI}'
                     + '文化程度是${文化程度}'
                     + '幼年时家庭经济状况是${幼年时家庭经济状况}'
                     + '揉眼睛的频率是${揉眼睛的频率}'
                     + '每次揉眼持续时间是${每次揉眼持续时间}'
                     + '揉眼时的力度是${揉眼时的力度}'
                     + '幼年时居住地是${幼年时居住地}')
template_yiduo_02 = (
        'K1 F (D) is ${K1 F (D)}, '
        + 'K2 F (D) is ${}, '
        + 'Km F (D) is ${}, '
        + 'Maximum keratometry of the front surface is ${}, '
        + 'Steepest point of the front surface keratometry displacement in the x-axis  is ${}, '
        + 'Steepest point of the front surface keratometry displacement in the y-axis  is ${}, '
        + 'RMS (CF)  is ${}, '
        + 'RMS HOA (CF)  is ${}, '
        + 'RMS LOA (CF)  is ${}, '
        + 'K1 B (D)  is ${}, '
        + 'K2 B (D)  is ${}, '
        + 'Km B (mm)  is ${}, '
        + 'RMS (CB)  is ${}, '
        + 'RMS HOA (CB)  is ${}, '
        + 'RMS LOA (CB)  is ${}, '
        + 'Pachy Apex(CCT)  is ${}, '
        + 'Pachy Min (TCT)  is ${}, '
        + 'Dist. Apex-Thin.Loc. [mm](Dist. C-T)  is ${}, '
        + 'X position of the thinnest point  is ${}, '
        + 'Y position of the thinnest point  is ${}, '
        + 'Mean eccentricity in the central 30 degrees by Fourier analysis  is ${}, '
        + 'Root-mean-square of total aberrations of whole cornea  is ${}, '
        + 'Root-mean-square of higher-order aberrations of whole cornea  is ${}, '
        + 'Root-mean-square of lower-order aberrations of whole cornea  is ${}, '
        + 'Corneal volume in a 3mm diameter zone around the corneal apex  is ${}, '
        + 'Corneal volume in a 5mm diameter zone around the corneal apex  is ${}, '
        + 'Corneal volume in a 7mm diameter zone around the corneal apex  is ${}, '
        + 'Corneal volume in a 10mm diameter zone around the corneal apex  is ${}, '
        + 'BAD Df  is ${}, '
        + 'BAD Db  is ${}, '
        + 'BAD Dp  is ${}, '
        + 'BAD Dt  is ${}, '
        + 'BAD Da  is ${}, '
        + 'BAD Dy  is ${}, '
        + 'Index of surface variance  is ${}, '
        + 'index of vertical asymmetry  is ${}, '
        + 'Keratoconus index  is ${}, '
        + 'Central keratoconus index  is ${}, '
        + 'Index of height asymmetry  is ${}, '
        + 'Index of height decentration  is ${}, '
        + 'Minimum radius of curvature  is ${}, '
        + 'Pachy Prog Index Min.  is ${}, '
        + 'Pachy Prog Index Max.  is ${}, '
        + 'Pachy Prog Index Avg.  is ${}, '
        + 'Minimum Ambrósio relational thickness  is ${}, '
        + 'Maximum Ambrósio relational thickness  is ${}, '
        + 'Average Ambrósio relational thickness  is ${}, '
        + 'C.Volume(chamber volume)  is ${}, '
        + 'Chamber angle  is ${}, '
        + 'A.C.Depth Int  is ${}, '
        + 'Elevation front  is ${}, '
        + 'Elevation back  is ${}, ')
template_yiduo_3 = ('Intraocular pressure  is ${}, '
                    + 'Maximum deformation amplitude  is ${}, '
                    + 'Time of reaching the first applanation  is ${}, '
                    + 'Velocity of the corneal apex at the first applanation  is ${}, '
                    + 'Time of reaching the second applanation  is ${}, '
                    + 'Velocity of the corneal apex at the second applanation  is ${}, '
                    + 'Time of undergoing the greatest degree of deformation and reaching the highest concavity  is ${}, '
                    + 'Distance between the two bending peaks created in the cornea at the highest concavity  is ${}, '
                    + 'Length at the the first applanation  is ${}, '
                    + 'Highest concavity deflection length  is ${}, '
                    + 'Length at the the second applanation  is ${}, '
                    + 'Deflection amplitude of the corneal apex at the first applanation  is ${}, '
                    + 'Deflection amplitude of the corneal apex at the highest concavity  is ${}, '
                    + 'Deflection amplitude of the corneal apex at the second applanation  is ${}, '
                    + 'Deflection area between the initial convex cornea and cornea at the first applanation on the analyzed horizontal sectional plane  is ${}, '
                    + 'Deflection area between the initial convex cornea and cornea at the highest concavity on the analyzed horizontal sectional plane  is ${}, '
                    + 'Deflection area between the initial convex cornea and cornea at the second applanation on the analyzed horizontal sectional plane  is ${}, '
                    + 'Change in Arclength during the first applanation moment from the initial state  is ${}, '
                    + 'Change in Arclength during the highest concavity moment from the initial state  is ${}, '
                    + 'Change in Arclength during the second applanation moment from the initial state  is ${}, '
                    + 'Maximum inverse concave radius  is ${}, '
                    + 'Ratio between the central deformation and the average of the peripheral deformation determined at 2mm  is ${}, '
                    + 'Ratio between the central deformation and the average of the peripheral deformation determined at 1mm  is ${}, '
                    + 'Ambrósio’s relational thickness in the horizontal profile  is ${}, '
                    + 'Biomechanically corrected intraocular pressure  is ${}, '
                    + 'Integrated radius  is ${}, '
                    + 'Stiffness parameter at the first applanation  is ${}, '
                    + 'Stress-strain index  is ${}, '
                    + 'label'
                    )
