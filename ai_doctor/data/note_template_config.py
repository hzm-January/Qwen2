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
DSMTR_GT = "gt"
DSMTR_LT = "lt"
TYPE_DIGIT = 0
TYPE_WORD = 1
TYPE_WORD_P = 2  # 需要特殊处理
TYPE_WORD_P2 = 3  # 需要特殊处理choices values

rule_yiduo = {
    # "":{},
    # 性别	年龄	BMI	文化程度	幼年时家庭经济状况	揉眼睛的频率	每次揉眼持续时间	揉眼时的力度	最常揉眼的部位
    # 揉眼姿势	最常采用的睡姿	睡觉时是否打鼾或患有睡眠呼吸暂停综合征？	春季角结膜炎	过敏性结膜炎	倒睫	干眼症
    # 眼睑松弛综合征	是否患有过敏性疾病？	是否对某些物质过敏？	甲状腺疾病	是否患有其他疾病？	是否用过外源性性激素药物？
    # 职业	幼年时居住地	睡觉时是否偏好把手或手臂垫放在眼睛上？	每天使用电子屏幕（手机、电脑等）的总时间（小时）
    # 每天在黑暗环境中使用电子屏幕的时间（小时）	阅读书籍	每天在户外阳光/紫外线下活动时间（小时）	常在大量灰尘环境中工作或生活？
    # 常于夜间工作/学习？	感到工作/学习压力很大？	是否吸烟？	是否饮酒？	是否怀过孕？	惯用手	圆锥角膜家族史
    "性别": {"type": TYPE_WORD, "note": "本患者性别{value}"},
    "年龄": {"type": TYPE_WORD, "note": "年龄{value}岁"},
    "BMI": {"type": TYPE_DIGIT, "bins": [INT_MIN, 24, INT_MAX], "labels": [FLAG_NORM, FLAG_ABNORM], "discriminator": DSMTR_LT, "threshold": 24},
    "文化程度": {"type": TYPE_WORD, "note": "{value}文化程度"},
    "幼年时家庭经济状况": {"type": TYPE_WORD, "note": "幼年时家庭经济状况{value}"},
    "揉眼睛的频率": {"type": TYPE_WORD_P2, "choices": ["总是", "经常", "有时", "偶尔"], "values": ["1天大于10次", "1天4~10次", "1天1~3次", "少于1天1次"], "note": "{choice}揉眼睛，揉眼睛的频率大约为{value}"},
    "每次揉眼持续时间": {"type": TYPE_WORD_P, "choices": ["<10秒", "≥10秒"], "values": ["小于10秒", "大于10秒", "1天1~3次", "少于1天1次"], "note": "每次揉眼睛持续时间{value}"},
    "揉眼时的力度": {"type": TYPE_WORD_P, "choices": ["轻", "中", "重"], "values": ["较轻", "适中", "较重"],  "note": "揉眼时的力度{value}"},
    "最常揉眼的部位": {"type": TYPE_WORD_P, "choices": ["眼角", "无差异", "眼睑"], "values": ["较轻", "适中", "较重"],  "note": "揉眼时的力度{value}"},
    "揉眼姿势": {"type": TYPE_WORD_P, "choices": ["指尖", "指关节", "全指尖", "掌根"], "values": ["最轻", "较轻", "适中", "较重"],  "note": "常使用{choice}姿势揉眼，该姿势力度{value}"},
    "最常采用的睡姿": {"type": TYPE_WORD, "note": "最常采用{value}的睡姿"},
    "睡觉时是否打鼾或患有睡眠呼吸暂停综合征？": {"type": TYPE_WORD_P, "choices": ["是", "否"], "values": ["打鼾或患有", "不打鼾也没有"], "note": "睡觉时{value}睡眠呼吸暂停综合征"},
    "春季角结膜炎": {"type": TYPE_WORD_P, "choices": ["是", "否"], "values": ["患有", "没有"], "note": "{value}春季角结膜炎"},
    "过敏性结膜炎": {"type": TYPE_WORD_P, "choices": ["是", "否"], "values": ["患有", "没有"], "note": "{value}过敏性结膜炎"},
    "倒睫": {"type": TYPE_WORD_P, "choices": ["是", "否"], "values": ["患有", "没有"], "note": "{value}倒睫"},
    "干眼症": {"type": TYPE_WORD_P, "choices": ["是", "否"], "values": ["患有", "没有"], "note": "{value}干眼症"},
    "眼睑松弛综合征": {"type": TYPE_WORD_P, "choices": ["是", "否"], "values": ["患有", "没有"], "note": "{value}眼睑松弛综合征"},
    "是否患有过敏性疾病？": {"type": TYPE_WORD_P, "choices": ["是", "否"], "values": ["患有", "没有"], "note": "{value}过敏性疾病"},
    "是否对某些物质过敏？": {"type": TYPE_WORD_P, "choices": ["是", "否"], "values": ["对某些物质过敏", "没有物质过敏史"], "note": "{value}"},
    "甲状腺疾病": {"type": TYPE_WORD_P, "choices": ["是", "否"], "values": ["患有", "没有"], "note": "{value}甲状腺疾病"},
    "是否患有其他疾病？": {"type": TYPE_WORD_P, "choices": ["是", "否"], "values": ["患有", "没有"], "note": "{value}其他疾病"},
    "是否用过外源性性激素药物？": {"type": TYPE_WORD_P, "choices": ["是", "否"], "values": ["", "没有"], "note": "{value}用过外源性性激素药物"},
    "职业": {"type": TYPE_WORD_P, "choices": ["学生", "无业", "在办公室工作", "非办公室工作"], "values": ["该患者是一名学生", "该患者无业或者已退休", "该患者在办公室工作", "该患者不在办公室工作"], "note": "{value}"},
    "幼年时居住地": {"type": TYPE_WORD, "note": "幼年在{value}居住"},
    "睡觉时是否偏好把手或手臂垫放在眼睛上？": {"type": TYPE_WORD_P, "choices": ["是", "否"], "values": ["偏好", "不会"], "note": "睡觉时{value}把手或手臂垫放在眼睛上"},
    "每天使用电子屏幕（手机、电脑等）的总时间（小时）": {"type": TYPE_WORD, "note": "每天使用电子屏幕（手机、电脑等）的总时间为{value}小时"},
    "每天在黑暗环境中使用电子屏幕的时间（小时）": {"type": TYPE_WORD, "note": "每天在黑暗环境中使用电子屏幕的时间为{value}小时）"},
    "阅读书籍": {"type": TYPE_WORD, "note": "每天阅读书籍的时间为{value}小时）"},
    "每天在户外阳光/紫外线下活动时间（小时）": {"type": TYPE_WORD, "note": "每天在户外阳光/紫外线下活动时间为{value}小时）"},
    "常在大量灰尘环境中工作或生活？": {"type": TYPE_WORD_P, "choices": ["是", "否"], "values": ["经常", "不常"], "note": "{value}在大量灰尘环境中工作或生活"},
    "常于夜间工作/学习？": {"type": TYPE_WORD_P, "choices": ["是", "否"], "values": ["经常", "不常"], "note": "{value}在大量灰尘环境中工作或生活"},
    "感到工作/学习压力很大？": {"type": TYPE_WORD_P, "choices": ["是", "否"], "values": ["", "没有"], "note": "{value}感到工作/学习压力很大"},
    "是否吸烟？": {"type": TYPE_WORD_P, "choices": ["是", "否"], "values": ["", "不"], "note": "{value}吸烟"},
    "是否饮酒？": {"type": TYPE_WORD_P, "choices": ["是", "否"], "values": ["", "不"], "note": "{value}饮酒"},
    "是否怀过孕？": {"type": TYPE_WORD_P, "choices": ["是", "否"], "values": ["", "没有"], "note": "{value}怀过孕"},
    "惯用手": {"type": TYPE_WORD, "note": "{value}为惯用手"},
    "圆锥角膜家族史": {"type": TYPE_WORD_P, "choices": ["是", "否"], "values": ["有", "没有"], "note": "{value}圆锥角膜家族史"},

    "K1 F (D)": {"type": TYPE_DIGIT, "bins": [INT_MIN, 45, INT_MAX], "labels": [FLAG_NORM, FLAG_ABNORM], "discriminator": DSMTR_LT, "threshold": 45},
    "K2 F (D)": {"type": TYPE_DIGIT, "bins": [INT_MIN, 49, INT_MAX], "labels": [FLAG_NORM, FLAG_ABNORM], "discriminator": DSMTR_LT, "threshold": 49},
    "Km F (D)": {"type": TYPE_DIGIT, "bins": [INT_MIN, 46, INT_MAX], "labels": [FLAG_NORM, FLAG_ABNORM], "discriminator": DSMTR_LT, "threshold": 45},
    "K Max (Front)": {"type": TYPE_DIGIT, "bins": [INT_MIN, 48.2, INT_MAX], "labels": [FLAG_NORM, FLAG_ABNORM], "discriminator": DSMTR_LT, "threshold": 48.2},
    "K Max X (Front)": {"type": TYPE_DIGIT, "bins": [INT_MIN, 0.07, INT_MAX], "labels": [FLAG_NORM, FLAG_ABNORM], "discriminator": DSMTR_LT, "threshold": 0.07},
    "K Max Y (Front)": {"type": TYPE_DIGIT, "bins": [INT_MIN, -1, INT_MAX], "labels": [FLAG_ABNORM, FLAG_NORM], "discriminator": DSMTR_GT, "threshold": -1},
    "RMS (CF)": {"type": TYPE_DIGIT, "bins": [INT_MIN, 2, INT_MAX], "labels": [FLAG_NORM, FLAG_ABNORM], "discriminator": DSMTR_LT, "threshold": 2},
    "RMS HOA (CF)": {"type": TYPE_DIGIT, "bins": [INT_MIN, 0.4, INT_MAX], "labels": [FLAG_NORM, FLAG_ABNORM], "discriminator": DSMTR_LT, "threshold": 0.4},
    "RMS LOA (CF)": {"type": TYPE_DIGIT, "bins": [INT_MIN, 2, INT_MAX], "labels": [FLAG_NORM, FLAG_ABNORM], "discriminator": DSMTR_LT, "threshold": 2},

    "K1 B (D)": {"type": TYPE_DIGIT, "bins": [INT_MIN, -6.4, INT_MAX], "labels": [FLAG_ABNORM, FLAG_NORM], "discriminator": DSMTR_GT, "threshold": -6.4},
    "K2 B (D)": {"type": TYPE_DIGIT, "bins": [INT_MIN, -6.6, INT_MAX], "labels": [FLAG_ABNORM, FLAG_NORM], "discriminator": DSMTR_GT, "threshold": -6.6},
    "Km B (mm)": {"type": TYPE_DIGIT, "bins": [INT_MIN, -6.4, INT_MAX], "labels": [FLAG_ABNORM, FLAG_NORM], "discriminator": DSMTR_GT, "threshold": -6.4},

    "RMS (CB)": {"type": TYPE_DIGIT, "bins": [INT_MIN, 1, INT_MAX], "labels": [FLAG_NORM, FLAG_ABNORM], "discriminator": DSMTR_LT, "threshold": -6.4},
    "RMS HOA (CB)": {"type": TYPE_DIGIT, "bins": [INT_MIN, 0.25, INT_MAX], "labels": [FLAG_NORM, FLAG_ABNORM], "discriminator": DSMTR_LT, "threshold": -6.4},
    "RMS LOA (CB)": {"type": TYPE_DIGIT, "bins": [INT_MIN, 1.1, INT_MAX], "labels": [FLAG_NORM, FLAG_ABNORM], "discriminator": DSMTR_LT, "threshold": -6.4},

    "Pachy Apex:(CCT)": {"type": TYPE_DIGIT, "bins": [INT_MIN, 516, INT_MAX], "labels": [FLAG_ABNORM, FLAG_NORM], "discriminator": DSMTR_GT, "threshold": -6.4},
    "Pachy Min: (TCT)": {"type": TYPE_DIGIT, "bins": [INT_MIN, 511, INT_MAX], "labels": [FLAG_ABNORM, FLAG_NORM], "discriminator": DSMTR_GT, "threshold": -6.4},

    "Dist. Apex-Thin.Loc. [mm](Dist. C-T)": {"type": TYPE_DIGIT, "bins": [INT_MIN, 0.9, INT_MAX],
                                              "labels": [FLAG_NORM, FLAG_ABNORM], "discriminator": DSMTR_LT, "threshold": 0.9},
    "PachyMinX": {},
    "PachyMinY": {},
    "EccSph": {"type": TYPE_DIGIT, "bins": [INT_MIN, 0.6, INT_MAX], "labels": [FLAG_NORM, FLAG_ABNORM], "discriminator": DSMTR_LT, "threshold": 0.6},
    "RMS (Cornea)": {"type": TYPE_DIGIT, "bins": [INT_MIN, 2, INT_MAX], "labels": [FLAG_NORM, FLAG_ABNORM], "discriminator": DSMTR_LT, "threshold": 2},
    "RMS HOA (Cornea)": {"type": TYPE_DIGIT, "bins": [INT_MIN, 0.5, INT_MAX], "labels": [FLAG_NORM, FLAG_ABNORM], "discriminator": DSMTR_LT, "threshold": 0.5},
    "RMS LOA (Cornea)": {"type": TYPE_DIGIT, "bins": [INT_MIN, 2, INT_MAX], "labels": [FLAG_NORM, FLAG_ABNORM], "discriminator": DSMTR_LT, "threshold": 2},

    "C.Vol D 3mm": {},
    "C.Vol D 5mm": {},
    "C.Vol D 7mm": {},
    "C.Vol D 10mm": {},
    "BAD Df": {"type": TYPE_DIGIT, "bins": [INT_MIN, 2.6, INT_MAX], "labels": [FLAG_NORM, FLAG_ABNORM], "discriminator": DSMTR_LT, "threshold": 2.6},
    "BAD Db": {"type": TYPE_DIGIT, "bins": [INT_MIN, 2.6, INT_MAX], "labels": [FLAG_NORM, FLAG_ABNORM], "discriminator": DSMTR_LT, "threshold": 2.6},
    "BAD Dp": {"type": TYPE_DIGIT, "bins": [INT_MIN, 2.6, INT_MAX], "labels": [FLAG_NORM, FLAG_ABNORM], "discriminator": DSMTR_LT, "threshold": 2.6},
    "BAD Dt": {"type": TYPE_DIGIT, "bins": [INT_MIN, 2.6, INT_MAX], "labels": [FLAG_NORM, FLAG_ABNORM], "discriminator": DSMTR_LT, "threshold": 2.6},
    "BAD Da": {"type": TYPE_DIGIT, "bins": [INT_MIN, 2.6, INT_MAX], "labels": [FLAG_NORM, FLAG_ABNORM], "discriminator": DSMTR_LT, "threshold": 2.6},
    "BAD Dy": {"type": TYPE_DIGIT, "bins": [INT_MIN, 2.6, INT_MAX], "labels": [FLAG_NORM, FLAG_ABNORM], "discriminator": DSMTR_LT, "threshold": 2.6},
    "ISV": {"type": TYPE_DIGIT, "bins": [INT_MIN, 41, INT_MAX], "labels": [FLAG_NORM, FLAG_ABNORM], "discriminator": DSMTR_LT, "threshold": 41},
    "IVA": {"type": TYPE_DIGIT, "bins": [INT_MIN, 0.32, INT_MAX], "labels": [FLAG_NORM, FLAG_ABNORM], "discriminator": DSMTR_LT, "threshold": 0.32},
    "KI": {"type": TYPE_DIGIT, "bins": [INT_MIN, 1.07, INT_MAX], "labels": [FLAG_NORM, FLAG_ABNORM], "discriminator": DSMTR_LT, "threshold": 1.07},
    "CKI": {"type": TYPE_DIGIT, "bins": [INT_MIN, 1.03, INT_MAX], "labels": [FLAG_NORM, FLAG_ABNORM], "discriminator": DSMTR_LT, "threshold": 1.03},
    "IHA": {"type": TYPE_DIGIT, "bins": [INT_MIN, 21, INT_MAX], "labels": [FLAG_NORM, FLAG_ABNORM], "discriminator": DSMTR_LT, "threshold": 21},
    "IHD": {"type": TYPE_DIGIT, "bins": [INT_MIN, 0.016, INT_MAX], "labels": [FLAG_NORM, FLAG_ABNORM], "discriminator": DSMTR_LT, "threshold": 0.016},
    "R Min (mm)": {"type": TYPE_DIGIT, "bins": [INT_MIN, 7, INT_MAX], "labels": [FLAG_ABNORM, FLAG_NORM], "discriminator": DSMTR_GT, "threshold": 7},
    "Pachy Prog Index Min.": {"type": TYPE_DIGIT, "bins": [INT_MIN, 0.88, INT_MAX], "labels": [FLAG_NORM, FLAG_ABNORM], "discriminator": DSMTR_LT, "threshold": 0.88},
    "Pachy Prog Index Max.": {"type": TYPE_DIGIT, "bins": [INT_MIN, 1.58, INT_MAX], "labels": [FLAG_NORM, FLAG_ABNORM], "discriminator": DSMTR_LT, "threshold": 1.58},
    "Pachy Prog Index Avg.": {"type": TYPE_DIGIT, "bins": [INT_MIN, 1.08, INT_MAX], "labels": [FLAG_NORM, FLAG_ABNORM], "discriminator": DSMTR_LT, "threshold": 1.08},
    "ART Min.": {},
    "ART Max.": {"type": TYPE_DIGIT, "bins": [INT_MIN, 412, INT_MAX], "labels": [FLAG_ABNORM, FLAG_NORM], "discriminator": DSMTR_GT, "threshold": 412},
    "ART Avg.": {},
    "C.Volume(chamber volume)": {"type": TYPE_DIGIT, "bins": [INT_MIN, 210, INT_MAX], "labels": [FLAG_NORM, FLAG_ABNORM], "discriminator": DSMTR_LT, "threshold": 210},
    "Chamber angle": {"type": TYPE_DIGIT, "bins": [INT_MIN, 44, INT_MAX], "labels": [FLAG_NORM, FLAG_ABNORM], "discriminator": DSMTR_LT, "threshold": 44},
    "A.C.Depth Int": {"type": TYPE_DIGIT, "bins": [INT_MIN, 3.4, INT_MAX], "labels": [FLAG_NORM, FLAG_ABNORM], "discriminator": DSMTR_LT, "threshold": 3.4},
    "Elevation front": {"type": TYPE_DIGIT, "bins": [INT_MIN, 6, INT_MAX], "labels": [FLAG_NORM, FLAG_ABNORM], "discriminator": DSMTR_LT, "threshold": 6},
    "Elevation back": {"type": TYPE_DIGIT, "bins": [INT_MIN, 12.3, INT_MAX], "labels": [FLAG_NORM, FLAG_ABNORM], "discriminator": DSMTR_LT, "threshold": 12.3},
    "IOP [mmHg]": {},
    "DA [mm]": {},
    "A1T [ms]": {"type": TYPE_DIGIT, "bins": [INT_MIN, 7.15, INT_MAX], "labels": [FLAG_ABNORM, FLAG_NORM], "discriminator": DSMTR_GT, "threshold": 7.15},
    "A1V [m/s]": {"type": TYPE_DIGIT, "bins": [INT_MIN, 0.16, INT_MAX], "labels": [FLAG_NORM, FLAG_ABNORM], "discriminator": DSMTR_LT, "threshold": 0.16},
    "A2T [ms]": {"type": TYPE_DIGIT, "bins": [INT_MIN, 22.1, INT_MAX], "labels": [FLAG_NORM, FLAG_ABNORM], "discriminator": DSMTR_LT, "threshold": 22.1},
    "A2V [m/s]": {"type": TYPE_DIGIT, "bins": [INT_MIN, -0.3, INT_MAX], "labels": [FLAG_ABNORM, FLAG_NORM], "discriminator": DSMTR_GT, "threshold": -0.3},
    "HCT [ms]": {"type": TYPE_DIGIT, "bins": [INT_MIN, 17.25, INT_MAX], "labels": [FLAG_NORM, FLAG_ABNORM], "discriminator": DSMTR_LT, "threshold": 17.25},
    "PD [mm]": {"type": TYPE_DIGIT, "bins": [INT_MIN, 5.3, INT_MAX], "labels": [FLAG_NORM, FLAG_ABNORM], "discriminator": DSMTR_LT, "threshold": 5.3},
    "A1L [mm]": {},
    "HCDL [mm]": {},
    "A2L [mm]": {},
    "A1DeflAmp. [mm](A1DA)": {"type": TYPE_DIGIT, "bins": [INT_MIN, 0.095, INT_MAX], "labels": [FLAG_NORM, FLAG_ABNORM], "discriminator": DSMTR_LT, "threshold": 0.095},
    "HCDeflAmp. [mm](HCDA)": {"type": TYPE_DIGIT, "bins": [INT_MIN, 1, INT_MAX], "labels": [FLAG_NORM, FLAG_ABNORM], "discriminator": DSMTR_LT, "threshold": 1},
    "A2DeflAmp. [mm](A2DA)": {"type": TYPE_DIGIT, "bins": [INT_MIN, 0.11, INT_MAX], "labels": [FLAG_NORM, FLAG_ABNORM], "discriminator": DSMTR_LT, "threshold": 0.11},
    "A1DeflArea [mm^2]": {"type": TYPE_DIGIT, "bins": [INT_MIN, 0.19, INT_MAX], "labels": [FLAG_NORM, FLAG_ABNORM], "discriminator": DSMTR_LT, "threshold": 0.19},
    "HCDeflArea [mm^2]": {"type": TYPE_DIGIT, "bins": [INT_MIN, 4, INT_MAX], "labels": [FLAG_NORM, FLAG_ABNORM], "discriminator": DSMTR_LT, "threshold": 4},
    "A2DeflArea [mm^2]": {"type": TYPE_DIGIT, "bins": [INT_MIN, 0.25, INT_MAX], "labels": [FLAG_NORM, FLAG_ABNORM], "discriminator": DSMTR_LT, "threshold": 0.25},
    "A1ΔArcL [mm]": {},
    "HCΔArcL [mm]": {},
    "A2ΔArcL [mm]": {},
    "MaxIR [mm^-1]": {"type": TYPE_DIGIT, "bins": [INT_MIN, 0.18, INT_MAX], "labels": [FLAG_NORM, FLAG_ABNORM], "discriminator": DSMTR_LT, "threshold": 0.18},
    "DAR2": {"type": TYPE_DIGIT, "bins": [INT_MIN, 5, INT_MAX], "labels": [FLAG_NORM, FLAG_ABNORM], "discriminator": DSMTR_LT, "threshold": 5},
    "DAR1": {"type": TYPE_DIGIT, "bins": [INT_MIN, 1.55, INT_MAX], "labels": [FLAG_NORM, FLAG_ABNORM], "discriminator": DSMTR_LT, "threshold": 1.55},
    "ARTh": {"type": TYPE_DIGIT, "bins": [INT_MIN, 300, INT_MAX], "labels": [FLAG_ABNORM, FLAG_NORM], "discriminator": DSMTR_GT, "threshold": 300},
    "bIOP": {"type": TYPE_DIGIT, "bins": [INT_MIN, 14, INT_MAX], "labels": [FLAG_ABNORM, FLAG_NORM], "discriminator": DSMTR_GT, "threshold": 14},
    "IR [mm^-1]": {"type": TYPE_DIGIT, "bins": [INT_MIN, 9, INT_MAX], "labels": [FLAG_NORM, FLAG_ABNORM], "discriminator": DSMTR_LT, "threshold": 9},
    "SP A1": {"type": TYPE_DIGIT, "bins": [INT_MIN, 80, INT_MAX], "labels": [FLAG_ABNORM, FLAG_NORM], "discriminator": DSMTR_GT, "threshold": 80},
    "SSI": {"type": TYPE_DIGIT, "bins": [INT_MIN, 0.75, INT_MAX], "labels": [FLAG_ABNORM, FLAG_NORM], "discriminator": DSMTR_GT, "threshold": 0.75},
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
