from kfp import dsl
from ..runner.local_runner import LocalDagRunner

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


def test_local_dag_runner():
    # Initialize the runner
    runner = LocalDagRunner()

    # Define parameters
    payload = {
        "recipient": "World",
    }

    # Run the pipeline
    result = runner.run(hello_pipeline, payload)
    
    # Check results
    assert result.output == "Hello, World!"