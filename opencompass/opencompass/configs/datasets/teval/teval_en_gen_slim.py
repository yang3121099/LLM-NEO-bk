"""Slim T-Eval config: only review subtask."""
from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import ChatInferencer
from opencompass.openicl.icl_evaluator import TEvalEvaluator
from opencompass.datasets import teval_postprocess, TEvalDataset

teval_subject_mapping = {
    'review': ['review_str_v1'],
}

teval_reader_cfg = dict(input_columns=['prompt'], output_column='ground_truth')

teval_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(
            round=[
                dict(role='HUMAN', prompt='{prompt}'),
            ],
        ),
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=ChatInferencer),
)

teval_slim_datasets = []
for _name in teval_subject_mapping:
    teval_eval_cfg = dict(
        evaluator=dict(type=TEvalEvaluator, subset=_name),
        pred_postprocessor=dict(type=teval_postprocess),
        num_gpus=1,
    )
    for subset in teval_subject_mapping[_name]:
        teval_slim_datasets.append(
            dict(
                abbr='teval-' + subset,
                type=TEvalDataset,
                path='./data/teval/EN',
                name=subset,
                reader_cfg=teval_reader_cfg,
                infer_cfg=teval_infer_cfg,
                eval_cfg=teval_eval_cfg,
            )
        )
