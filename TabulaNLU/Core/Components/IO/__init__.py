import pickle
import tflearn
import os
import Core
from pathlib import Path
from datetime import datetime
def SaveTFModel(model, metadata, intents, project='default'):
    path = get_project_path()

    date = datetime.now().strftime('%Y%m%d-%H%M%S')

    model_path = os.path.join(path, 'models', project, 'model_' + date)

    #save model
    model.save(os.path.join(model_path,'/model.tflearn'))

    #save metadata 
    pickle.dump({'words':metadata.words, 'classes':metadata.classes, 'train_x':metadata.train_x, 'train_y':metadata.train_y}, open( os.path.join(model_path,'/training_data'), 'wb' ))

    #save intents
    file = open(os.path.join(model_path,'intents.json'), w)
    file.writelines(intents)
    file.close()

def load_tf_model(model, model_name = '', project_name = 'default'):
   path = get_project_path()

def get_project_path():
   path = Path(os.path.abspath(Core))
   root_path = Path(path.parent)
   return root_path.resolve()

   