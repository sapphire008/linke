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
    return df.to_dict("records")

def tfrecord_processing_fn(inputs: List[Dict], config={}):
    import numpy as np
    import pandas as pd
    from pdb import set_trace
    
    inputs["A"] = np.stack(inputs["A"])
    outputs = {}
    outputs["A"] = list(map(lambda x: [str(x[0]).encode()], inputs["A"])) # int -> byte
    outputs["B"] = list(map(lambda x: [float(x[0].decode("utf-8"))], inputs["B"])) # byte -> float
    outputs["C"] = list(map(lambda x: [int(x[0])], inputs["C"]))
    # Convert to list of dict before returning
    outputs = pd.DataFrame(outputs).to_dict("records")
    return outputs