from transformers import AutoTokenizer, AutoModelForCausalLM, logging
import torch
import json
from tqdm.auto import tqdm
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import pairwise_distances
from argparse import ArgumentParser
import os

logging.set_verbosity_warning()

STOP = []
SURE = []
UNSURE = []

# Hyperparameters for clustering
n_clusters = 3
m = 2.0  # Fuzziness parameter for FCM

# Model and tokenizer configuration
hg_token = '<Your API Token>'
cache_dir = '/data3/hg_weight/hg_weight/'

def inference(input_text):
    full_input = f"Question: {input_text} Answer:"
    inputs = tokenizer(full_input, return_tensors="pt").to("cuda")
    ids = inputs['input_ids']
    outputs = model.generate(
        ids,
        max_new_tokens=5,
        output_scores=True,
        return_dict_in_generate=True
    )
    logits = outputs['scores']
    output_sequence = []
    product = torch.tensor(1.0, device='cuda')
    count = 0
    for i in logits:
        pt = torch.softmax(torch.Tensor(i[0]), dim=0)
        max_loc = torch.argmax(pt)
        if max_loc in STOP:
            break
        else:
            output_sequence.append(max_loc)
            product *= torch.max(pt)
            count += 1

    output_text = tokenizer.decode(output_sequence) if output_sequence else ""
    average_confidence = np.power(product.item(), (1/count)).item() if count > 0 else 0.0
    return output_text, average_confidence

def checksure(input_text):
    full_input = f"{input_text}. Are you sure you accurately answered the question based on your internal knowledge? I am"
    inputs = tokenizer(full_input, return_tensors="pt").to("cuda")
    ids = inputs['input_ids']
    outputs = model.generate(
        ids,
        max_new_tokens=1,
        output_scores=True,
        return_dict_in_generate=True
    )
    logits = outputs['scores']
    pt = torch.softmax(torch.Tensor(logits[0][0]), dim=0)
    sure_prob = pt[SURE[0]]
    unsure_prob = pt[UNSURE[0]]
    normalized_sure_prob = sure_prob / (sure_prob + unsure_prob) if (sure_prob + unsure_prob) > 0 else 0.0
    return normalized_sure_prob.item()

class FuzzyCMeans:
    def __init__(self, n_clusters=3, m=2.0, max_iter=100, error=1e-5):
        self.n_clusters = n_clusters
        self.m = m
        self.max_iter = max_iter
        self.error = error

    def fit(self, X):
        N = X.shape[0]
        U = np.random.dirichlet(np.ones(self.n_clusters), size=N)
        for iteration in range(self.max_iter):
            U_old = U.copy()
            centers = self._calculate_cluster_centers(X, U)
            U = self._update_membership_matrix(X, centers)
            if np.linalg.norm(U - U_old) < self.error:
                break
        self.U = U
        self.centers = centers

    def _calculate_cluster_centers(self, X, U):
        um = U ** self.m
        centers = (um.T @ X) / um.sum(axis=0)[:, None]
        return centers

    def _update_membership_matrix(self, X, centers):
        dist = pairwise_distances(X, centers)
        inv_dist = 1.0 / dist
        power = 2 / (self.m - 1)
        U_new = inv_dist ** power
        U_new /= U_new.sum(axis=1, keepdims=True)
        return U_new

    def predict(self, X):
        return np.argmax(self.U, axis=1)

def classify_by_membership(prob, centers):
    dists = [abs(prob - c) for c in centers]
    min_dist_idx = np.argmin(dists)
    categories = ['low', 'mid', 'high']
    return categories[min_dist_idx]

def plot_fuzzy_c_means(sure_probs, centers, clusters, title, save_path='fcm_visualization.png'):
    plt.figure(figsize=(8, 6))
    for cluster in range(centers.shape[0]):
        points = sure_probs[clusters == cluster]
        plt.scatter(points, np.zeros_like(points) + cluster, label=f'Cluster {cluster}')
    plt.scatter(centers, np.zeros_like(centers), marker='x', s=200, c='red', label='Centers')
    plt.title(title)
    plt.xlabel("Probability")
    plt.yticks([])
    plt.legend()
    plt.grid()
    plt.savefig(save_path)
    #plt.show()


def fever_format_question(input_data):

    choices = ["A", "B", "C"]
    candidate_answer = ['SUPPORTS.','REFUTES.','NOT ENOUGH INFO.']
    
    evidence = " ".join(input_data["evidence"])
    full_input = "Evidence:" + evidence + "\nClaim:" + input_data['claim'] + "\nQuestion:" + "Does the evidence support the claim?" 
    for i in range(len(choices)):
        full_input += '\n' + choices[i] + ': ' + candidate_answer[i]
    full_input += "\nAnswer:" 

    return full_input

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--dataset', type=str, default="  ")
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--model_name', type=str, required=True)
    parser.add_argument('--model_size', type=str, required=True)
    parser.add_argument('--result', type=str, default="FEVER")
    parser.add_argument('--domain', type=str, default="_")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True, unk_token="<unk>", bos_token="<s>", eos_token="</s>", add_bos_token=False, token=hg_token, cache_dir=cache_dir)
    model = AutoModelForCausalLM.from_pretrained(args.model, device_map='auto', token=hg_token, cache_dir=cache_dir)

    STOP.append(tokenizer(".")['input_ids'][0])
    SURE.append(tokenizer("sure")['input_ids'][0])
    UNSURE.append(tokenizer("unsure")['input_ids'][0])

    data = []

    if args.dataset=='pararel':
        with open(f"/data3/KJE/code/WIL_DeepLearningProject_2/NS_LLM/data/pararel/training_data.json", 'r') as f:
            data = json.load(f)
    elif args.dataset=='hotpot_10k':
        with open(f"/data3/KJE/code/WIL_DeepLearningProject_2/NS_LLM/data/HotpotQA/{args.dataset}.json",'r') as f:
            data = json.load(f)
    elif args.dataset=='hotpot_10k':
        with open(f"/data3/KJE/code/WIL_DeepLearningProject_2/NS_LLM/data/HotpotQA/{args.dataset}.json",'r') as f:
            data = json.load(f)

    results = []
    predict_conf_list = []
    sure_probs_list = []

    for sample in tqdm(data):
        if args.dataset=='pararel':
            question = sample[0]
            answer = sample[1]
        elif args.dataset=='hotpot_10k':
            question = sample
            answer = sample['answer']

        elif args.dataset=='fever_10k':
            mapping = {'SUPPORTS':"A",'REFUTES':"B",'NOT ENOUGH INFO':"C"}
            question = sample
            answer = mapping[sample['label']]


        output, predict_conf = inference(question)
        sure_prob = checksure(f"Question: {question} Answer: {output}")

        predict_conf_list.append([predict_conf])
        sure_probs_list.append([sure_prob])
        # print(f"Generated confidence: {predict_conf}, Sure Probability: {sure_prob}")

        if answer in output:
            results.append((1, predict_conf, sure_prob))
        else:
            results.append((0, predict_conf, sure_prob))

        torch.cuda.empty_cache()

    # `predict_conf` 클러스터링 수행
    predict_conf_array = np.array(predict_conf_list)
    fcm_predict_conf = FuzzyCMeans(n_clusters=n_clusters, m=m, max_iter=100, error=1e-5)
    fcm_predict_conf.fit(predict_conf_array)

    # `sure_prob` 클러스터링 수행
    sure_probs_array = np.array(sure_probs_list)
    fcm_sure_prob = FuzzyCMeans(n_clusters=n_clusters, m=m, max_iter=100, error=1e-5)
    fcm_sure_prob.fit(sure_probs_array)

    # 클러스터링 결과 시각화 및 저장
    np.savez(f'results/version1/{args.model_name}/{args.model_size}/{args.result}_{args.domain}_fcm_predict_conf_results.npz', centers=fcm_predict_conf.centers, membership_matrix=fcm_predict_conf.U, clusters=fcm_predict_conf.predict(predict_conf_array))
    plot_fuzzy_c_means(predict_conf_array, fcm_predict_conf.centers, fcm_predict_conf.predict(predict_conf_array), title="Predict Confidence Clustering", save_path='fcm_predict_conf_visualization.png')

    np.savez(f'results/{args.result}_{args.domain}_fcm_sure_prob_results.npz', centers=fcm_sure_prob.centers, membership_matrix=fcm_sure_prob.U, clusters=fcm_sure_prob.predict(sure_probs_array))
    plot_fuzzy_c_means(sure_probs_array, fcm_sure_prob.centers, fcm_sure_prob.predict(sure_probs_array), title="Sure Probability Clustering", save_path='fcm_sure_prob_visualization.png')

    # 최종 결과 저장
    with open(f"results/{args.result}_{args.domain}.json", 'w') as f:
        json.dump(results, f)
