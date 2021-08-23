import os
import pickle
import tempfile

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