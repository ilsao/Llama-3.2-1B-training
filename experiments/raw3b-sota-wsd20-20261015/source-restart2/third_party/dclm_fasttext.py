"""DCLM's scoring function, copied unchanged except for its type annotation.

Source: https://github.com/mlfoundations/dclm/blob/
361714bdd60bb9b7f4b2d8354cebbf0dec0c329e/baselines/mappers/enrichers/
quality_prediction_enrichers_calc_fasttext.py
See DCLM-LICENSE.txt for the upstream MIT license.
"""


def classify_fasttext_hq_prob(model, content: str) -> dict:
    '''
    This function classifies a given text as either 'CC' or 'Wikipedia' and returns the label along with its probability.

    Parameters:
        model (fasttext.FastText._FastText): The FastText model to use for the classification.
        content (str): The text to classify.

    Returns:
        dict: A value for 'hq_prob' - the probability to be a high-quality page.
    '''
    output = {}
    text = " ".join(content.strip().splitlines())
    pred = model.predict(text)
    (pred_label, pred_prob) = pred
    pred_label = pred_label[0]
    hq_prob = pred_prob[0]
    if pred_label == "__label__cc":
        hq_prob = 1 - hq_prob
    return hq_prob
