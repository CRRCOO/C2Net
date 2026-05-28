import py_sod_metrics as metrics
import numpy as np


class EvaluationMetrics():
    def __init__(self):
        self.SM = metrics.Smeasure()
        self.EM = metrics.Emeasure()
        self.FM = metrics.Fmeasure()
        # self.WFM = metrics.WeightedFmeasure()
        self.MAE = metrics.MAE()

    def reset(self):
        self.__init__()

    def step(self, pred, gt):
        """
        pred: [0, 255], prediction maps (already sigmoid and times 255)
        gt: [0, 255]
        """
        self.SM.step(pred=pred, gt=gt)
        self.EM.step(pred=pred, gt=gt)
        self.FM.step(pred=pred, gt=gt)
        # self.WFM.step(pred=pred, gt=gt)
        self.MAE.step(pred=pred, gt=gt)

    def get_results(self):
        # S-measure, default alpha=0.5
        sm = self.SM.get_results()["sm"]
        # mean E-measure
        emMean = self.EM.get_results()["em"]['curve'].mean()
        # adaptive E-measure
        emAdp = self.EM.get_results()["em"]['adp']
        # F-measure
        fm = self.FM.get_results()["fm"]['curve'].mean()
        # weighted F-measure
        # wfm = self.WFM.get_results()["wfm"]
        # mean Absolute Error
        mae = self.MAE.get_results()["mae"]
        # return sm, emMean, emAdp, wfm, mae
        return {
            'sm': sm,
            'emMean': emMean,
            'emAdp': emAdp,
            'fm': fm,
            'mae': mae
        }
