from typing import Dict

def processing_fn(inputs, config={}):
    import pandas as pd
    df = pd.DataFrame(inputs)
    df = df[["A", "B"]]
    return df.to_dict("records")


def init_fn() -> Dict:
    return {"a": "asdf", "b": "bsdf"}