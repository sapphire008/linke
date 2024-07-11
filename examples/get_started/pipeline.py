
from kfp import dsl, compiler, local
from kfp.client import Client


# Starting a local runner
# Setting use_venv = False so that it runs on the current environment
local.init(
    runner=local.SubprocessRunner(use_venv=False),
    pipeline_root="./local_outputs",
    raise_on_error=True,
)

@dsl.component
def say_hello(name: str) -> str:
    hello_text = f'Hello, {name}!'
    print(hello_text)
    return hello_text

# Create the pipeline
@dsl.pipeline
def hello_pipeline(recipient: str)-> str:
    hello_task = say_hello(name=recipient)
    return hello_task.output

# Run locally
pipeline_task = hello_pipeline(recipient="World")
print(pipeline_task.output)
assert pipeline_task.output == "Hello, World!"


# Compile the pipeline
# compiler.Compiler().compile(hello_pipeline, package_path='pipeline.yaml')
# Run the pipeline remotely
# client = Client(host='host-name')
# run = client.create_run_from_pipeline_package(
#     'pipeline.yaml',
#     arguments={
#         'recipient': 'World',
#     }
# )
