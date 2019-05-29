import pickle
import tflearn
import os
import Core
import json
import tensorflow as tf
from pathlib import Path
from datetime import datetime
def SaveTFModel(model, metadata, intents, project='default'):
    path = get_project_path()

    date = datetime.now().strftime('%Y%m%d-%H%M%S')
    model_name = 'model_' + date
    model_path = os.path.join(path, 'models', project, 'model_' + date)

    #save model
    model.save(os.path.join(model_path,'/model.tflearn'))

    #save metadata
    pickle.dump({'words':metadata.words, 'classes':metadata.classes, 'train_x':metadata.train_x, 'train_y':metadata.train_y}, open(os.path.join(model_path,'/training_data'), 'wb'))

    #save intents
    file = open(os.path.join(model_path,'intents.json'), w)
    file.writelines(intents)
    file.close()
    return model_name

def load_tf_model(model_name='', project_name='default'):
    model_path = get_model_path()

    data = pickle.load(open(os.path.join(model_path,'training_data'),'rb'))
    with open(os.path.join(model_path,'intents.json')) as json_data:
        intents = json.load(json_data)
        tf.reset_default_graph()
    # Build neural network
        net = tflearn.input_data(shape=[None, len(train_x[0])])
        net = tflearn.fully_connected(net, 8)
        net = tflearn.fully_connected(net, 8)
        net = tflearn.fully_connected(net, len(train_y[0]), activation='softmax')
        net = tflearn.regression(net)

        # Define model and setup tensorboard
        model = tflearn.DNN(net, tensorboard_dir=os.path.join(get_project_path(),'tflearn_logs'))

        model.load(os.path.join())

        return {data,intents,model}

def get_model_path():
    path = get_project_path()

    project_path = os.path.join(path,'models',project_name)
    possible_path = os.path.join(project_path,model_name)

    if os.path.isdir(possible_path): 
        return possible_path
    else:
        all_subdirs = [d for d in os.listdir(project_path) if os.path.isdir(d)]
        latest_subdir = max(all_subdirs, key=os.path.getmtime())
        return os.path.join(project_path,latest_subdir)

def get_project_path():
   path = Path(os.path.abspath(Core))
   root_path = Path(path.parent)
   return root_path.resolve()

   