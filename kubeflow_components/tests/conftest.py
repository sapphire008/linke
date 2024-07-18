from typing import Dict, List

def csv_processing_fn(inputs: List[Dict], config={}):
    import pandas as pd
    df = pd.DataFrame(inputs)
    df = df[["A", "B"]]
    return df.to_dict("records")


def csv_init_fn() -> Dict:
    return {"a": "asdf", "b": "bsdf"}


def bq_processing_fn(inputs: List[Dict], config={}):
    import pandas as pd
    df = pd.DataFrame(inputs)
    df["transformed_title"] = df["title"].apply(lambda x: x[:2])
    df["num_tags"] = df["tags"].apply(len)
    df.drop(columns=["title", "tags"], inplace=True)
    # from pdb import set_trace; set_trace()
    return df.to_dict("records")