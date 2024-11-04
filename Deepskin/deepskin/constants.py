#!/usr/bin/env python
# -*- coding: utf-8 -*-

import platform

__author__  = ['Nico Curti']
__email__ = ['nico.curti2@unibo.it']

__all__ = [
  'CRLF',
  'IMG_SIZE',
  'GREEN_COLOR_CODE',
  'ORANGE_COLOR_CODE',
  'VIOLET_COLOR_CODE',
  'RED_COLOR_CODE',
  'RESET_COLOR_CODE',
  'Deepskin_CENTER',
  'Deepskin_SCALE',
  'Deepskin_PWAT_PARAMS',
  'Deepskin_PWAT_BIAS',
]

IMG_SIZE = 256
GREEN_COLOR_CODE = '\033[38;5;40m'
ORANGE_COLOR_CODE = '\033[38;5;208m'
VIOLET_COLOR_CODE = '\033[38;5;141m'
RED_COLOR_CODE = '\033[38;5;196m'
RESET_COLOR_CODE = '\033[0m'
CRLF = '\r\x1B[K' if platform.system() != 'Windows' else '\r\x1b[2K'

Deepskin_CENTER = {
  'w_haralick0': 0.0002222729472489692,
  'w_haralick1': 1890.1645598237635,
  'w_haralick2': 0.39624571900533073,
  'w_haralick3': 1699.7507631374451,
  'w_haralick4': 0.05825257441765201,
  'w_haralick5': 165.46789722033162,
  'w_haralick6': 4859.627907301425,
  'w_haralick7': 7.907375987459552,
  'w_haralick8': 12.570986041853839,
  'w_haralick9': 8.417852836481736e-05,
  'w_haralick10': 5.955239259730355,
  'w_haralick11': -0.2021242003064579,
  'w_haralick12': 0.9619772153300806,
  'p_haralick0': 0.0001498540696012536,
  'p_haralick1': 1642.1494695401611,
  'p_haralick2': 0.5552398117391217,
  'p_haralick3': 1964.5852700529895,
  'p_haralick4': 0.06078474835571498,
  'p_haralick5': 205.18173777610355,
  'p_haralick6': 6119.316957254933,
  'p_haralick7': 8.187785208529927,
  'p_haralick8': 13.10517714959607,
  'p_haralick9': 8.513602245362374e-05,
  'p_haralick10': 5.943323499089892,
  'p_haralick11': -0.20929446749325206,
  'p_haralick12': 0.9697884563910389,
  'w_avgR': 0.5212469820939212,
  'w_avgG': 0.27109229463921075,
  'w_avgB': 0.21401858655694464,
  'p_avgR': 0.5763267835303386,
  'p_avgG': 0.3564247033287137,
  'p_avgB': 0.3081093298977088,
  'w_stdR': 0.09237739554699534,
  'w_stdG': 0.10829318823674319,
  'w_stdB': 0.09195912180260984,
  'p_stdR': 0.10716071451169598,
  'p_stdG': 0.1401474853536956,
  'p_stdB': 0.13216593170513105,
  'w_avgH': 0.06552020427372157,
  'w_avgS': 0.5892058218823246,
  'w_avgV': 0.5212469820939212,
  'p_avgH': 0.09036986402266009,
  'p_avgS': 0.48847457610644596,
  'p_avgV': 0.5767915389925715,
  'w_stdH': 0.11715889282107367,
  'w_stdS': 0.11403082370522448,
  'w_stdV': 0.09260579562238144,
  'p_stdH': 0.1824778713170838,
  'p_stdS': 0.1440560768776456,
  'p_stdV': 0.10723952971997479,
  'w_park': 0.28480298580007396,
  'p_park': 0.22188344728430864,
  'w_amparo': 0.03164616239230859,
  'p_amparo': 0.03691944795994818,
}

Deepskin_SCALE = {
  'w_haralick0': 0.0002325046095526415,
  'w_haralick1': 1404.4257572010195,
  'w_haralick2': 0.23078393134467157,
  'w_haralick3': 965.2959521200373,
  'w_haralick4': 0.023780274040611858,
  'w_haralick5': 68.48109193307027,
  'w_haralick6': 2630.755286941684,
  'w_haralick7': 0.5066619026943684,
  'w_haralick8': 1.4678491082231115,
  'w_haralick9': 4.888152908641458e-05,
  'w_haralick10': 0.5095925355951518,
  'w_haralick11': 0.11575370812484267,
  'w_haralick12': 0.06070190816734167,
  'p_haralick0': 9.672668886112473e-05,
  'p_haralick1': 953.4214077362369,
  'p_haralick2': 0.21065800501136922,
  'p_haralick3': 925.506077994615,
  'p_haralick4': 0.018952140859000446,
  'p_haralick5': 57.92742976496601,
  'p_haralick6': 3173.260857014111,
  'p_haralick7': 0.38873009569827843,
  'p_haralick8': 0.8671021401509353,
  'p_haralick9': 3.675366174882123e-05,
  'p_haralick10': 0.39261737487581705,
  'p_haralick11': 0.062112428576971634,
  'p_haralick12': 0.03229774274624053,
  'w_avgR': 0.16656629897564573,
  'w_avgG': 0.14635937524567533,
  'w_avgB': 0.11880385258454706,
  'p_avgR': 0.13331435845295891,
  'p_avgG': 0.11832798552663454,
  'p_avgB': 0.13275959348198357,
  'w_stdR': 0.03928345907627907,
  'w_stdG': 0.045476869848503826,
  'w_stdB': 0.046438403018434535,
  'p_stdR': 0.04882281971219227,
  'p_stdG': 0.049730916792113555,
  'p_stdB': 0.05846673654613731,
  'w_avgH': 0.0928823575050199,
  'w_avgS': 0.16858715377732175,
  'w_avgV': 0.1666455216195598,
  'p_avgH': 0.1636084021024,
  'p_avgS': 0.15054938267363227,
  'p_avgV': 0.13390847886402646,
  'w_stdH': 0.18810319877291057,
  'w_stdS': 0.04483667865035873,
  'w_stdV': 0.0391485600808012,
  'p_stdH': 0.2047438529951371,
  'p_stdS': 0.05252145976774261,
  'p_stdV': 0.04903280080434655,
  'w_park': 0.15173598777434077,
  'p_park': 0.10460738864219321,
  'w_amparo': 0.0455553377351849,
  'p_amparo': 0.05808570032051939,
}

Deepskin_PWAT_PARAMS = {
  'w_haralick0': -0.22608690262627693,
  'w_haralick1': 0.8419020236399831,
  'w_haralick2': 1.9840516080509525,
  'w_haralick4': -0.014677772064076867,
  'w_haralick6': -0.028192574114897707,
  'w_haralick7': 0.033573325209037996,
  'w_haralick9': 0.4454964606178141,
  'w_haralick10': 0.6586148883190893,
  'w_haralick11': -0.13888628129503355,
  'p_haralick1': 0.17821612965307904,
  'p_haralick6': -0.010119483685282258,
  'p_haralick7': 0.327880140420109,
  'p_haralick9': -0.25511946678446584,
  'p_haralick10': 0.10776636887371593,
  'w_avgR': -0.11711651785031672,
  'w_avgG': 2.1350537406274075,
  'w_avgB': -0.2800676295856026,
  'p_avgR': -1.0264783767720724,
  'p_avgG': 0.7963529644104094,
  'w_stdG': -0.3488198067525097,
  'w_stdB': -1.073605999704211,
  'w_avgH': -0.7512504032512959,
  'w_avgV': -2.044720823209275,
  'p_avgH': 1.2681488355543602,
  'p_avgS': 1.1481222816655834,
  'w_stdH': -0.5847556821893894,
  'w_stdS': 0.02710634669502382,
  'w_stdV': 0.3894296265346003,
  'p_stdH': -0.05659291142960778,
  'p_stdS': 0.07895362196139402,
  'p_stdV': 0.11913258096905475,
  'w_park': -1.5749437194228606,
  'w_amparo': 0.0712150447907629,
  'p_amparo': 0.21322070485461567,
}

Deepskin_PWAT_BIAS = 14.244211150770264
