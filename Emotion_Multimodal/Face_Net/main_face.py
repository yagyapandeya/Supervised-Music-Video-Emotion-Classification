import os
import numpy as np
import pickle
import matplotlib.pyplot as plt
from scipy import interp
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score, roc_auc_score, roc_curve, auc

from face_net import video_audio_net

emo_classes = {'Excited': 0, 'Fear': 1, 'Neutral': 2, 'Relaxation': 3, 'Sad': 4, 'Tension': 5}

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

def create_folder(fd):
    if not os.path.exists(fd):
        os.makedirs(fd)

def normalize(img):
    '''
    Normalizes an array 
    (subtract mean and divide by standard deviation)
    '''
    eps = 0.001
    if np.std(img) != 0:
        img = (img - np.mean(img)) / np.std(img)
    else:
        img = (img - np.mean(img)) / eps
    return img

def get_classes(key, dict_genres):
    # Transforming data to help on transformation
    #labels = []
    tmp_genre = {v:k for k,v in dict_genres.items()}

    return tmp_genre[key]

def majority_voting(scores, dict_genres):
    preds = np.argmax(scores, axis = 1)
    values, counts = np.unique(preds, return_counts=True)
    counts = np.round(counts/np.sum(counts), 2)
    votes = {k:v for k, v in zip(values, counts)}
    votes = {k: v for k, v in sorted(votes.items(), key=lambda item: item[1], reverse=True)}
    return [(get_classes(x, dict_genres), prob) for x, prob in votes.items()]

Out_dir = "TEST_OUTCOME_dir/"
create_folder(Out_dir)

#Loading test dataset
fname_npz = "mv_test_db.npz"

loadeddata = np.load(fname_npz)   
#video3D_C3D, video3D_fast, video3D_slow, melphase, melphase_fast, melphase_slow, target = loadeddata["X_norm"], loadeddata["X_fast"], loadeddata["X_slow"], loadeddata["melphasegram"], loadeddata["melphasegram_fast"], loadeddata[#"melphasegram_slow"], loadeddata["labels"]

video3D_fast, target = loadeddata["face_fast"], loadeddata["target"]

video3D_fast = normalize(video3D_fast)
print("The video3D_fast data shape", video3D_fast.shape)

img_rows, img_cols, frames, vid_channel  = 128, 128, 64, 3
video_input_size = (img_rows, img_cols, frames, vid_channel)
audio_input_size = (128, 1292, 2)

network = video_audio_net(video_input_size)
print("The Network Weight Loaded")

loss, acc = network.evaluate(x=[video3D_fast], y=target, batch_size=10, verbose=0)
print('Test loss:', loss)
print('Test accuracy:', acc)

predictions = network.predict(x=[video3D_fast],batch_size=10, verbose=0)    #x=X_test

#Mejority voting
votes = majority_voting(predictions, emo_classes)
#print("{} is a {} song".format(val_generator[0], votes[0][0]))
print("most likely genres are: {}".format(votes[:10]))

def get_confusion_matrix_one_hot(model_results, truth):
    assert model_results.shape == truth.shape
    num_outputs = truth.shape[1]
    confusion_matrix = np.zeros((num_outputs, num_outputs), dtype=np.int32)
    predictions = np.argmax(model_results, axis=1)
    assert len(predictions) == truth.shape[0]
    
    for actual_class in range(num_outputs):
        idx_examples_this_class = truth[:, actual_class] == 1
        prediction_for_this_class = predictions[idx_examples_this_class]
        for predicted_class in range(num_outputs):
            count = np.sum(prediction_for_this_class == predicted_class)
            confusion_matrix[actual_class, predicted_class] = count            
    assert np.sum(confusion_matrix) == len(truth)
    assert np.sum(confusion_matrix) == np.sum(truth)
    
    return confusion_matrix
    
confusion_matrix = get_confusion_matrix_one_hot(predictions, target) #y_test
print(confusion_matrix)

#If you want to save prediction
with open(Out_dir + 'ROC.pkl', 'wb') as f:
    pickle.dump(predictions, f)


def plot_ROC():
    SM_pred_probs = predictions
    n_samples = np.min([len(SM_pred_probs)])
    
    def plot_roc_curves(y_true, pred_probs):
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        
        NUM_CLASSES = 6
        for i in range(6):
            fpr[i], tpr[i], _ = roc_curve(y_true[:, i], pred_probs[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])    
    
        # First aggregate all false positive rates
        all_fpr = np.unique(np.concatenate([fpr[i] for i in range(NUM_CLASSES)]))
    
        # Then interpolate all ROC curves at this points
        mean_tpr = np.zeros_like(all_fpr)
        for i in range(NUM_CLASSES):
            mean_tpr += interp(all_fpr, fpr[i], tpr[i])
    
        # Finally average it and compute AUC
        mean_tpr /= NUM_CLASSES
    
        return all_fpr, mean_tpr, auc(all_fpr, mean_tpr)
    
    # Plot all ROC curves
    plt.figure(figsize=(6, 6), dpi=300)
    #plt.figure(figsize=(10,9))
    
    plt.title('Macro-average ROC curves')
    
    fpr, tpr, roc_auc = plot_roc_curves(target[:n_samples], SM_pred_probs[:n_samples])
    plt.plot(fpr, tpr,label='Softmax Classifier (Area = {0:0.3f})'''.format(roc_auc), color='red', linestyle=':', linewidth=2)
    
    plt.plot([0, 1], [0, 1], 'k--', lw=1)
    plt.grid()
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    #plt.xlabel('False Positive Rate', fontsize=9.5, family ='Times New Roman') # Change fornt in specific line
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve of Music Video Emotion')  #Receiver Operating Characteristic (ROC) Curve
    plt.legend(loc="lower right")
    plt.savefig(Out_dir + 'roc-curve.png')
    plt.savefig(Out_dir + 'roc-curve.pdf')
    
    #plt.show()

print('Classifier result save in disk ')
plot_ROC()
print('Ploted ROC for multi-class')

print('Softmax Classifier ROC AUC score= {0:.3f}'.format(roc_auc_score(y_true=target, y_score=predictions, average='macro')))
print('Softmax Classifier Test Set F1-score =  {0:.2f}'.format(f1_score(target, np.round(predictions), average='macro')))


####################################PLOT CONFUSION MATRIX####################################################################
from sklearn import metrics
import seaborn as sns

LABELS = ["Exciting",
          "Fear",
          "Neutral",
          "Relaxation",
          "Sad",
          "Tension"]

def show_confusion_matrix(validations, predictions):
    matrix = metrics.confusion_matrix(validations, predictions)
    #plt.figure(figsize=(6, 4), dpi = 300)    #ORG
    plt.figure(figsize=(7, 6), dpi = 300)
    sns.heatmap(matrix,
                cmap="coolwarm",
                linecolor='white',
                linewidths=1,
                xticklabels=LABELS,
                yticklabels=LABELS,
                annot=True,
                fmt="d")
    plt.title("Confusion Matrix", fontsize=15)
    plt.ylabel("True Label", fontsize=10)
    plt.xlabel("Predicted Label", fontsize=10)
    plt.savefig(Out_dir + 'Confusion_matrix.png')
    plt.savefig(Out_dir + 'Confusion_matrix.pdf')
    #plt.show()

print("\n--- Confusion matrix for test data ---\n")

# Take the class with the highest probability from the test predictions
max_y_pred_test = np.argmax(predictions, axis=1)
max_y_test = np.argmax(target, axis=1)

show_confusion_matrix(max_y_test, max_y_pred_test)

