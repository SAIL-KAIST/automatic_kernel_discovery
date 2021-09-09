import os
import pickle
import tempfile
import mlflow
from mlflow.utils.file_utils import get_local_path_or_none, TempDir
from mlflow.tracking.artifact_utils import _download_artifact_from_uri
import itertools

def load_pickle(file):
    
    with open(file, "rb") as f:
        result = pickle.load(f)
    
    return result

def save(obj, name):
    
    tempdir = tempfile.mkdtemp()
    save_file = os.path.join(tempdir, name)
    with open(save_file, "wb") as f:
        pickle.dump(obj, f)
    
    print(f"Save an obj to temporary file {save_file}")
    
    return save_file

def download_artifacts(path, storage_dir):
    local_path = get_local_path_or_none(path)
    if local_path:
        if not os.path.exists(local_path):
            raise ValueError(f"Cannot find artifact locally at {local_path}")
        return os.path.abspath(local_path)
    
    base_name = os.path.basename(path)
    
    download_dir = os.path.join(storage_dir, base_name)
    return _download_artifact_from_uri(path, download_dir)

def get_search_and_analysis(client, run):
    
    experiment_id = run.info.experiment_id
    children_runs = client.search_runs([experiment_id], f"tags.mlflow.parentRunId = '{run.info.run_id}'") 
    
    if run.data.tags.__contains__("search_run") and run.data.tags.__contains__("analysis_run"):
        # if there are tags
        search_run_id =  run.data.tags["search_run"]
        analysis_run_id = run.data.tags["analysis_run"]
        search_run = client.get_run(run_id=search_run_id)
        analysis_run = client.get_run(run_id=analysis_run_id)
    else:
        # no tag, find by entrypoint
        for child in children_runs:
            if child.data.tags['mlflow.project.entryPoint'] == 'search_kernel':
                search_run = child
                print(f"Found search run: {search_run.info.run_id}")
            elif child.data.tags["mlflow.project.entryPoint"] == 'analysis':
                analysis_run = child
                print(f"Found analysis run: {analysis_run.info.run_id}")
    
    return search_run, analysis_run

def export(run_id, track_uri=None, search_id=None, analysis_id=None):
    """ Export MLflow run to the format of XAI website

    Args:
        run_id (str): Run ID
        track_uri (str, optional): Tracking URI. Defaults to None.
    """
    
    if track_uri:
        mlflow.set_tracking_uri(track_uri)
    client = mlflow.tracking.MlflowClient()
    
    run = client.get_run(run_id=run_id)
    
    if search_id and analysis_id:
        search_run = client.get_run(run_id=search_id)
        analysis_run = client.get_run(run_id=analysis_id)
    else:
        search_run, analysis_run = get_search_and_analysis(client, run)
    
    component_uri = os.path.join(analysis_run.info.artifact_uri, "components.pkl")
    
    component_file = download_artifacts(component_uri, storage_dir=None)
    components = load_pickle(component_file)
    
    search_artifacts = client.list_artifacts(run_id=search_run.info.run_id)
    analysis_artifacts = client.list_artifacts(run_id=analysis_run.info.run_id)

    search_artifacts = [os.path.join(search_run.info.artifact_uri, a.path)  for a in search_artifacts]
    analysis_artifacts = [os.path.join(analysis_run.info.artifact_uri, a.path)  for a in analysis_artifacts]
    
    
    print("Image artifacts: ")
    images = []
    for a in itertools.chain(search_artifacts, analysis_artifacts):
        ext = os.path.splitext(a)[-1].lower()
        if ext in [".png"]:
            print("\t", a)
            images.append(a)
    
    with TempDir() as temp:
        components = download_artifacts(component_file, temp.path)
        client.log_artifact(run_id, components)
        for image in images:
            dir = download_artifacts(image, temp.path)
            client.log_artifact(run_id, dir)
    
    print("component log artifact")