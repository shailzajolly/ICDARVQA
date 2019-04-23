import torch
import pickle
import nltk
import numpy as np

class VqaMetric():

    def __init__(self, fpath):
        self.idx2ans, _ = pickle.load(open(fpath, 'rb'))


    def get_real_idxs(self,decoder_input, mca):
        """

        Args:
            decoder_input: 1D Torch tensor (batch_size)
            mca: Torch tensor (batch_size, 104)

        Returns:
        """

        real_idxs = torch.gather(mca, decoder_input.unsqueeze(1))
        return real_idxs.squeeze()


    def get_ans_from_idxs(self, ans_idxs):

        output_ans = []

        for time_step in ans_idxs:
            words = [self.idx2ans[ans_idx.item()] for ans_idx in time_step]
            output_ans.append(words)

        ans_sents = [" ".join(sent).replace("PAD","").replace("EOS", "").strip()
                     for sent in zip(*output_ans)]

        return ans_sents


    def icdar_metric(self, pred, gts):
        scores = []
        for gt in gts:
            score = 1 - (nltk.edit_distance(pred, gt) / float(max(len(pred), len(gt))))
            if score >= 0.5:
                scores.append(score)
            else:
                scores.append(0.0)

        return max(scores)


    def compute_score(self, decoder_ans_idxs, ans_txt):

        predictions = self.get_ans_from_idxs(decoder_ans_idxs)

        scores = []
        for pred, gts in zip(predictions, ans_txt):
            score = self.icdar_metric(pred, gts)
            scores.append(score)

        return np.mean(scores)