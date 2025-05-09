import os
import torch

import numpy as np
import torch.nn as nn
from huggingface_hub import PyTorchModelHubMixin
from transformers import EncoderDecoderModel, RobertaTokenizerFast, PreTrainedModel
from torch.utils.data import DataLoader, TensorDataset


class DependencyAnalyzer(nn.Module, PyTorchModelHubMixin):
    def __init__(self, encoder: PreTrainedModel = None,
                 match_tokenizer: RobertaTokenizerFast = None):
        super(DependencyAnalyzer, self).__init__()
        if not encoder:
            encoder: PreTrainedModel = EncoderDecoderModel.from_encoder_decoder_pretrained(
                "microsoft/codebert-base", "microsoft/codebert-base").encoder
        if match_tokenizer:
            encoder.resize_token_embeddings(len(match_tokenizer))
            encoder.config.decoder_start_token_id = match_tokenizer.cls_token_id
            encoder.config.pad_token_id = match_tokenizer.pad_token_id
            encoder.config.eos_token_id = match_tokenizer.sep_token_id
            encoder.config.vocab_size = match_tokenizer.vocab_size
        self.encoder = encoder
        self.dense = nn.Linear(768, 2)

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask)
        pooler_output = outputs.pooler_output
        output_2d = self.dense(pooler_output)
        return output_2d


def load_model_and_tokenizer():
    model_dir = os.path.join(
        os.path.dirname(__file__),
        '..',
        '..',
        '..',
        'models',
        'dependency-analyzer')
    model_path = os.path.join(model_dir, 'pytorch_model.bin')

    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    tokenizer = RobertaTokenizerFast.from_pretrained('microsoft/codebert-base')
    special_tokens = ['<from>', '<to>']
    tokenizer.add_tokens(special_tokens, special_tokens=True)
    model = DependencyAnalyzer(match_tokenizer=tokenizer)
    model.load_state_dict(torch.load(model_path, map_location=device))
    return model, tokenizer


class DependencyClassifier:
    def __init__(self):
        self.model, self.tokenizer = load_model_and_tokenizer()
        if torch.cuda.is_available():
            self.model.to(torch.device('cuda'))
        elif torch.backends.mps.is_available():
            self.model.to(torch.device('mps'))

    def construct_pair(self, code_1: str, code_2: str):
        return '<from>' + code_1 + '<to>' + code_2

    def construct_corpus_pair(self, corpus: 'list[tuple[str, str]]'):
        return [self.construct_pair(code_1, code_2)
                for code_1, code_2 in corpus]

    def gen(self, text: str):
        sigmoid = nn.Sigmoid()
        # ATTENTION: converted to batch here
        token_input = self.tokenizer(text, return_tensors='pt')
        if torch.cuda.is_available():
            token_input = token_input.to(torch.device('cuda'))
        elif torch.backends.mps.is_available():
            token_input = token_input.to(torch.device('mps'))

        with torch.no_grad():
            outputs = self.model(
                input_ids=token_input['input_ids'],
                attention_mask=token_input['attention_mask']
            )[0]
        outputs = sigmoid(outputs).detach().cpu()
        return outputs[1]

    def batch_gen(self, corpus_pair: 'list[str]'):
        sigmoid = nn.Sigmoid()
        if torch.cuda.is_available():
            device = torch.device('cuda')
        elif torch.backends.mps.is_available():
            device = torch.device('mps')
        else:
            device = torch.device('cpu')
        token_input = self.tokenizer(
            corpus_pair,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=512)
        dataset = TensorDataset(
            token_input["input_ids"],
            token_input["attention_mask"])
        dataloader = DataLoader(dataset, batch_size=32, shuffle=False)

        preds = []
        with torch.no_grad():
            for batch in dataloader:
                batch_input, attention_mask = [
                    item.to(device) for item in batch]
                outputs = self.model(
                    input_ids=batch_input,
                    attention_mask=attention_mask)
                outputs = sigmoid(outputs)[:, 1]
                preds.append(outputs.detach().cpu())
        preds = torch.cat(preds, dim=0)
        return preds.numpy()


def cal_dep_score(hunk: dict, file_content: str,
                  dependency_analyzer: DependencyClassifier):
    def split2window_str(lines):
        windows = []
        for i in range(len(lines) // 10 + 1):
            if i == len(lines) // 10:
                window = ''.join(lines[i * 10:])
            else:
                window = ''.join(lines[i * 10:(i + 1) * 10])
            windows.append(window)
        return windows

    fileB_lines = file_content.splitlines()
    # split file lines into code windows (10 lines)
    hunk_window_str = ''.join(hunk['code_window'])
    code_window_strsB = split2window_str(fileB_lines)
    if len(code_window_strsB) == 0:
        print('failed to split fileB into windows')
        raise KeyError('failed to split fileB into windows')
    # form code windows pairs
    code_window_pairs = []
    for windowB in code_window_strsB:
        code_window_pairs.append((hunk_window_str, windowB))
    corpus_pair = dependency_analyzer.construct_corpus_pair(code_window_pairs)
    results = dependency_analyzer.batch_gen(corpus_pair)
    assert len(results) == len(code_window_strsB)
    # get dep score
    dep_score_max = np.max(results).item()
    dep_score_mean = np.mean(results).item()
    dep_score_min = np.min(results).item()
    dep_score_median = np.median(results).item()
    dep_score_std = np.std(results).item()
    return [dep_score_max, dep_score_mean,
            dep_score_min, dep_score_median, dep_score_std]
