import os
import torch
import pickle
import jsonlines
import numpy as np
from torch.utils.data import DataLoader
from sklearn.linear_model import LinearRegression
from transformers import RobertaTokenizer, RobertaModel
from .dependency_analyzer import cal_dep_score, DependencyClassifier
from .siamese_net import evaluate_embedding_model, load_siamese_data
from perf import Stopwatch
from model_manager import load_model_with_cache

MODEL_ROLE = "embedding"
OUTPUT_MAX = 10


def construct_discriminator_dataset(
        hunk, file_name_contents, dependency_analyzer):
    dataset = []
    for file_name_content in file_name_contents:
        dep_score_list = cal_dep_score(
            hunk, file_name_content[1], dependency_analyzer)
        sample = {}
        sample['hunk'] = hunk
        sample['file'] = file_name_content[1]
        sample['file_path'] = file_name_content[0]
        sample['dependency_score'] = [dep_score_list[0]]
        sample['label'] = 1
        dataset.append(sample)
    return dataset


def load_model(model_path):
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    model = RobertaModel.from_pretrained("huggingface/CodeBERTa-small-v1")
    tokenizer = RobertaTokenizer.from_pretrained(
        "huggingface/CodeBERTa-small-v1")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    return model, tokenizer, device


def load_reg_model(lang):
    # The regression model is fit based on the validation set
    with open(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'models', f"./{lang}/reg_model.pickle"), 'rb') as file:
        reg = pickle.load(file)
    
    return reg


class DiscriminatorPredictor:
    _instance = None
    _model = None
    _tokenizer = None
    _device = None
    _reg_model = None
    _dependency_analyzer = None

    def __init__(self):
        raise RuntimeError('Call get_instance() instead')

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls.__new__(cls)
            cls._instance._initialize()
        return cls._instance

    def _initialize(self):
        """初始化所有需要的模型和组件"""
        print("初始化判别器...")
        # 加载嵌入模型
        self._model, self._tokenizer, self._device = load_model_with_cache(
            MODEL_ROLE, 'python', load_model)
        # 加载回归模型
        self._reg_model = load_reg_model('python')
        # 加载依赖分析器
        self._dependency_analyzer = DependencyClassifier()
        print("判别器初始化完成")

    def predict(self, json_input):
        """预测方法"""
        stopwatch = Stopwatch()
        stopwatch.start()

        # 0. remove targetFilePath from input["files"]
        if (len(json_input["prevEdits"]) == 0):
            return {"data": []}
        prev_edit = json_input["prevEdits"][-1]
        prev_edit_hunk = {}
        prev_edit_hunk["code_window"] = [
            prev_edit["codeAbove"],
            prev_edit["beforeEdit"],
            prev_edit["codeBelow"]]

        for i in range(len(json_input["files"])):
            if json_input["files"][i][0] == json_input["targetFilePath"]:
                json_input["files"].pop(i)
                break
        if (len(json_input["files"]) == 0):
            return {"data": []}

        # 1. construct discriminator dataset
        dataset = construct_discriminator_dataset(
            prev_edit_hunk, json_input["files"], self._dependency_analyzer)
        stopwatch.lap('build code collection')

        # 2. Calculate the semantic similarity
        tensor_dataset = load_siamese_data(dataset, self._tokenizer, False)
        dataloader = DataLoader(tensor_dataset, batch_size=1, shuffle=False)
        embedding_similiarity = evaluate_embedding_model(
            self._model, dataloader, "test")
        stopwatch.lap('calculate the semantic similarity')

        # 3. Use linear regression to predict label
        X_test = [dataset[idx]["dependency_score"] + [embedding_similiarity[idx]]
                  for idx in range(len(embedding_similiarity))]
        y_pred = self._reg_model.predict(X_test)
        y_pred = [1 if y > 0.5 else 0 for y in y_pred]

        files_pred = []
        for i in range(len(y_pred)):
            if y_pred[i] == 1:
                files_pred.append(json_input["files"][i][0])
        stopwatch.lap('infer result')

        # 4. prepare output
        output = {"data": []}
        for file in files_pred:
            output["data"].append(file)
            if len(output["data"]) >= OUTPUT_MAX:
                break
        stopwatch.lap('post-process result')
        
        print("+++ Discriminator profiling:")
        stopwatch.print_result()

        return output


def predict(json_input, language):
    '''
    Function: this is the interface between discriminator and VSCode extension
    Args:
        input: dictionary
            {
                "files":            list, [[relativeFilePath, fileContent], ...]
                "targetFilePath":   string, the relative path of the file to be edited
                "commitMessage":    string, the edit description
                "prevEdits":        list, [{"beforeEdit": string, "afterEdit": string, "codeAbove": string, "codeBelow": string}, ...]
            }
    Return:
        output: dictionary, contains chosen files' path and content
            {
                "data": [string], relative file paths that are probably related to target file
            }
    '''
    predictor = DiscriminatorPredictor.get_instance()
    return predictor.predict(json_input)
