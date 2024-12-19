import json
from safety_protocol import PromptSafetyProtocol
from sklearn.metrics import roc_curve
import numpy as np
import random
import sys

def get_data(dataset):
    json_file = f"/afs/crc.nd.edu/user/s/swebster/Private/SecureSoftwareEngineering/compose_json/{dataset}_full.json"
    with open(json_file) as f:
        return json.load(f)

def get_eer_threshold(y_true, y_score):
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    fnr = 1 - tpr
    eer_threshold = thresholds[np.nanargmin(np.absolute((fnr - fpr)))]

    return eer_threshold

def main():
    # Config
    rejection_mode = "none"
    rejection_threshold = 0.5

    safety_protocol = PromptSafetyProtocol(rejection_mode=rejection_mode, rejection_threshold=rejection_threshold, verbose="True")

    datasets = ["tensortrust", "tt_extraction", "train_deepset", "test_deepset"]

    truths = []
    score_proportional = []
    score_aggregate = []

    for dataset in datasets:

        data = get_data(dataset)
        responses = dict()

        for prompt_id,prompt_data in data.items():
            # Process prompt safety
            reject_proportional, reject_aggregate = safety_protocol.get_reject_scores(prompt_data)

            truths.append(prompt_data["label"])
            score_proportional.append(reject_proportional)
            score_aggregate.append(reject_aggregate)

    eer_proportional = get_eer_threshold(truths, score_proportional)
    eer_aggregate = get_eer_threshold(truths, score_aggregate)
    print(eer_proportional, eer_aggregate)

    n_correct = 0
    for truth,score in zip(truths, score_proportional):
        pred = 1 if score >= eer_proportional else 0
        if pred == truth:
            n_correct += 1
    print("proportional:",n_correct/len(truths))

    n_correct = 0
    for truth,score in zip(truths, score_aggregate):
        pred = 1 if score >= eer_aggregate else 0
        if pred == truth:
            n_correct += 1
    print("aggregate:",n_correct/len(truths))



if __name__ == "__main__":
    # verbose_safety_report = True if sys.argv[1] == "True" else False
    # file_tag = sys.argv[2]
    # main(verbose_safety_report, file_tag)
    main()