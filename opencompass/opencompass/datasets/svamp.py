import json

from datasets import Dataset

from opencompass.registry import LOAD_DATASET
from opencompass.utils import get_data_path

from .base import BaseDataset


@LOAD_DATASET.register_module()
class SVAMPDataset(BaseDataset):

    @staticmethod
    def load(path):
        path = get_data_path(path, local_mode=True)
        dataset = []
        with open(path, 'r', encoding='utf-8') as f:
            raw = f.read().strip()
        # Support both JSON array and JSONL formats
        if raw.startswith('['):
            items = json.loads(raw)
        else:
            items = [json.loads(line) for line in raw.splitlines() if line.strip()]
        for item in items:
            question = item['Body'] + ' ' + item['Question']
            answer = str(int(item['Answer']))
            dataset.append({'question': question, 'answer': answer})
        dataset = Dataset.from_list(dataset)
        return dataset
