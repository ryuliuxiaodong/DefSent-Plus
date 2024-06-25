import numpy as np
from numpy.linalg import norm
from scipy import stats

from defsentplus import DefSentPlus


def cos_sim(x, y):
    A = np.array(x)
    B = np.array(y)
    cosine = np.dot(A, B) / (norm(A) * norm(B))
    return cosine


def test_sts(encoder, pooling, senteval_root_path):
    dataset_names = ['sts12', 'sts13', 'sts14', 'sts15', 'sts16', 'sts-b', 'sick']

    all_scores = []

    for dataset_name in dataset_names:
        if dataset_name == 'sick':
            sent1_list = []
            sent2_list = []
            scores = []
            with open(senteval_root_path + "/data/downstream/SICK/SICK_test_annotated.txt") as f:
                _ = next(f)
                for line in f:
                    _, sentence1, sentence2, score, *_ = line.strip().split("\t")
                    sent1_list.append(sentence1)
                    sent2_list.append(sentence2)
                    scores.append(float(score))
            length = len(sent1_list)
        elif dataset_name == 'sts12':
            sent1_list = []
            sent2_list = []
            scores = []
            datasets = [
                "MSRpar",
                "MSRvid",
                "SMTeuroparl",
                "surprise.OnWN",
                "surprise.SMTnews",
            ]
            for dataset in datasets:
                gs = open(senteval_root_path + "/data/downstream/STS/STS12-en-test/STS.gs." + dataset + ".txt", encoding='utf-8')
                f = open(senteval_root_path + "/data/downstream/STS/STS12-en-test/STS.input." + dataset + ".txt", encoding='utf-8')
                for line_input, line_gs in zip(f, gs):
                    sentence1, sentence2, *_ = line_input.strip().split("\t")
                    if line_gs.strip() == "":
                        continue
                    sent1_list.append(sentence1)
                    sent2_list.append(sentence2)
                    scores.append(float(line_gs.strip()))
            length = len(sent1_list)
        elif dataset_name == 'sts13':
            sent1_list = []
            sent2_list = []
            scores = []

            datasets = ["FNWN", "headlines", "OnWN"]
            for dataset in datasets:
                gs = open(senteval_root_path + "/data/downstream/STS/STS13-en-test/STS.gs." + dataset + ".txt", encoding='utf-8')
                f = open(senteval_root_path + "/data/downstream/STS/STS13-en-test/STS.input." + dataset + ".txt", encoding='utf-8')
                for line_input, line_gs, *_ in zip(f, gs):
                    sentence1, sentence2 = line_input.strip().split("\t")
                    if line_gs.strip() == "":
                        continue
                    sent1_list.append(sentence1)
                    sent2_list.append(sentence2)
                    scores.append(float(line_gs.strip()))
            length = len(sent1_list)
        elif dataset_name == 'sts14':
            sent1_list = []
            sent2_list = []
            scores = []
            datasets = [
                "deft-forum",
                "deft-news",
                "headlines",
                "images",
                "OnWN",
                "tweet-news",
            ]
            for dataset in datasets:
                gs = open(senteval_root_path + "/data/downstream/STS/STS14-en-test/STS.gs." + dataset + ".txt", encoding='utf-8')
                f = open(senteval_root_path + "/data/downstream/STS/STS14-en-test/STS.input." + dataset + ".txt", encoding='utf-8')
                for line_input, line_gs in zip(f, gs):
                    sentence1, sentence2, *_ = line_input.strip().split("\t")
                    if line_gs.strip() == "":
                        continue
                    sent1_list.append(sentence1)
                    sent2_list.append(sentence2)
                    scores.append(float(line_gs.strip()))
            length = len(sent1_list)
        elif dataset_name == 'sts15':
            sent1_list = []
            sent2_list = []
            scores = []
            datasets = [
                "answers-forums",
                "answers-students",
                "belief",
                "headlines",
                "images",
            ]
            for dataset in datasets:
                gs = open(senteval_root_path + "/data/downstream/STS/STS15-en-test/STS.gs." + dataset + ".txt", encoding='utf-8')
                f = open(senteval_root_path + "/data/downstream/STS/STS15-en-test/STS.input." + dataset + ".txt", encoding='utf-8')
                for line_input, line_gs in zip(f, gs):
                    sentence1, sentence2, *_ = line_input.strip().split("\t")
                    if line_gs.strip() == "":
                        continue
                    sent1_list.append(sentence1)
                    sent2_list.append(sentence2)
                    scores.append(float(line_gs.strip()))
            length = len(sent1_list)
        elif dataset_name == "sts16":
            sent1_list = []
            sent2_list = []
            scores = []
            datasets = [
                "answer-answer",
                "headlines",
                "plagiarism",
                "postediting",
                "question-question",
            ]

            for dataset in datasets:
                gs = open(senteval_root_path + "/data/downstream/STS/STS16-en-test/STS.gs." + dataset + ".txt", encoding='utf-8')
                f = open(senteval_root_path + "/data/downstream/STS/STS16-en-test/STS.input." + dataset + ".txt", encoding='utf-8')
                for line_input, line_gs in zip(f, gs):
                    sentence1, sentence2, *_ = line_input.strip().split("\t")
                    if line_gs.strip() == "":
                        continue
                    sent1_list.append(sentence1)
                    sent2_list.append(sentence2)
                    scores.append(float(line_gs.strip()))
            length = len(sent1_list)
        elif dataset_name == "sts-b":
            sent1_list = []
            sent2_list = []
            scores = []

            datasets = ["sts-test.csv"]
            for dataset in datasets:
                f = open(senteval_root_path + "/data/downstream/STS/STSBenchmark/" + dataset, encoding='utf-8')
                for line in f:
                    _, _, _, _, score, sentence1, sentence2, *_ = line.strip().split("\t")
                    sent1_list.append(sentence1)
                    sent2_list.append(sentence2)
                    scores.append(float(score))
            length = len(sent1_list)

        cos_sim_scores = []

        sent1_representations = encoder.encode(sentences=sent1_list, pooling=pooling).tolist()
        sent2_representations = encoder.encode(sentences=sent2_list, pooling=pooling).tolist()

        for i in range(length):
            cos_sim_scores.append(cos_sim(sent1_representations[i], sent2_representations[i]))

        x = np.array(cos_sim_scores)
        y = np.array(scores)
        corr, p = stats.spearmanr(x, y)
        print(dataset_name + ": " + str(round(corr, 4)))
        all_scores.append(round(corr, 4))

    print("Average: " + str(round(sum(all_scores) / len(all_scores), 4)))

if __name__ == "__main__":
    # Available backbone model name is bert-base-uncased, bert-large-uncased, roberta-base, or roberta-large.
    encoder = DefSentPlus("RyuKT/DefSentPlus-bert-base-uncased", backbone_model_name="bert-base-uncased", device="cuda")

    # Available pooling is "cls", "mean", or "prompt"
    pooling = "prompt"

    # Enter the correct root path of senteval in your environment.
    senteval_root_path = "./SentEval-main"

    test_sts(encoder=encoder, pooling=pooling, senteval_root_path=senteval_root_path)