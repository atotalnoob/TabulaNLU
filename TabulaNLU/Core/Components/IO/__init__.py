import pickle
import tflearn
from datetime import datetime
def SaveTFModel(model, metadata, project='default'):
    date = datetime.now().strftime('%Y%m%d-%H%M%S')
    model.save('../../../models/' + project+'/model_' +date + '.tflearn')
    pickle.dump({'words':metadata.words, 'classes':metadata.classes, 'train_x':metadata.train_x, 'train_y':metadata.train_y}, open( '../../../models/'+project+'/training_data_'+date, 'wb' ))
    