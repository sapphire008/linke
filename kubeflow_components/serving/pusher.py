"""
Push model artifacts to various cloud services for serving
"""

from kubernetes import client, config
from kubernetes.stream import stream
import tempfile
import os

# Load the Kubernetes configuration
config.load_kube_config()  # For local kubectl config
# or config.load_incluster_config()  # If running inside a pod

# Create API client
v1 = client.CoreV1Api()

# Define the pod and namespace
pod_name = "model-store-pod"
namespace = (
    "default"  # Change this if your pod is in a different namespace
)


def exec_command(command):
    return stream(
        v1.connect_get_namespaced_pod_exec,
        pod_name,
        namespace,
        command=command,
        stderr=True,
        stdin=False,
        stdout=True,
        tty=False,
    )


# Create directories
exec_command(["mkdir", "-p", "/pv/model-store/"])
exec_command(["mkdir", "-p", "/pv/config/"])


# Function to copy file to pod
def copy_file_to_pod(local_path, pod_path):
    with open(local_path, "rb") as file:
        data = file.read()

    exec_command(["mkdir", "-p", os.path.dirname(pod_path)])

    with tempfile.NamedTemporaryFile() as temp_file:
        temp_file.write(data)
        temp_file.flush()

        client.CoreV1Api().read_namespaced_pod(
            name=pod_name, namespace=namespace
        )

        exec_command(
            ["cp", "/dev/stdin", pod_path],
            stdin=open(temp_file.name, "rb"),
        )


# Copy files
copy_file_to_pod(
    "squeezenet1_1.mar", "/pv/model-store/squeezenet1_1.mar"
)
copy_file_to_pod("mnist.mar", "/pv/model-store/mnist.mar")
copy_file_to_pod("config.properties", "/pv/config/config.properties")

print("Files copied successfully.")
