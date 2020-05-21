# -*- coding: utf-8 -*-
"""
Created on Wed Mar 18 17:06:20 2020

@author: Nico
"""


from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import keras
from keras import backend as K
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.optimizers import RMSprop
from keras.layers import Input, Dense, GRU, Embedding
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard, ReduceLROnPlateau

#vérification des version utilisées
tf.__version__
keras.__version__
keras.backend.tf.__version__

#importation des données et mise en forme d'un dataframe à l'aide de Pandas
dataframe = pd.read_csv("KAG_energydata_complete.csv", sep=",",  parse_dates=['date'],index_col="date")
print(dataframe.head(5))


"""
ajouter la colonne avec heure... n'est utile que si on veut faire des analyse avec 
le graphique en fonction du temps. Donc pour créer un modèle ça ne sert à rien.
"""

"""
#on ajoute des colonnes à la fin du fichier de données contenant séparément l'heure, jour...
dataframe['hour'] = dataframe.index.hour
dataframe['day_of_month'] = dataframe.index.day
dataframe['day_of_week'] = dataframe.index.dayofweek
dataframe['month'] = dataframe.index.month
"""
"""
ANALYSE GRAPHIQUE (dois-je le garder dans ce fichier ou alors je créer un autre
fichier à part comme ça a été fait pour le programme d'affichage du réseau?)
"""
"""
#créé un histogramme de la température intérieure 1
dataframe.hist('T1')
#graphique de l'évolution de la température T1 en fonction du temps
fig,(ax1)= plt.subplots(nrows=1)
fig.set_size_inches(18, 8)
sns.lineplot(x=dataframe.index, y="T1", data=dataframe);
#donne la moyenne des températures T1 par mois en fontion du temps
df_by_month = dataframe.resample('M').mean()
fig,(ax1)= plt.subplots(nrows=1)
fig.set_size_inches(18, 8)
sns.lineplot(x=df_by_month.index, y="T1", data=df_by_month, ax=ax1);
"""
"""Affiche l'évolution de la température T1 en fonction des heures et en fonction de soit le jour, le mois..."""
"""
fig,(ax1, ax2)= plt.subplots(nrows=2)
fig.set_size_inches(18, 28)

sns.pointplot(data=dataframe, x='hour', y='T1', ax=ax1)
sns.pointplot(data=dataframe, x='hour', y='T1', hue='month', ax=ax2)
"""


"""EXPLOITATION DES DONNES"""
#Retrait des données superflus pour entrainer le model plus rapidement
dataframe.drop(['Appliances','lights','RH_1','T2','T3','T4','T5','T6','T7','T8'
                ,'T9','RH_2','RH_3','RH_4','RH_5','RH_6','RH_7','RH_8','RH_9'
                ,'Press_mm_hg','RH_out','Visibility','Tdewpoint','rv1','rv2']
                ,axis='columns', inplace=True)
dataframe.head()
#réduction de la taille du dataframe pour augmenter la vitesse d'entrainement
#on ne prend plus que 40% des données de base.
"""A VIRER LORSQUE L'ON EFFECTUERA L'ENTRAINEMENT FINAL"""
#dataframe_size = int(len(dataframe) * 0.4)
#dataframe = dataframe.iloc[0:dataframe_size]

"""shift des données pour pouvoir prédire la température 1h dans le future"""
"""
On veut pouvoir prédire la température dans 1h. Nous avons des échantillons de
la température toutes les 10min donc il faut décaler les données de 6 times_steps.
Pour pouvoir prédire le futur, il faut faire un décalage négatif.
"""
#nom du paramètre à déterminer
target_names = ['T1']
#on veut prédire la température dans 1h
shift_hours = 1
# Number of minutes (il y a des données toutes les 10 minutes donc dans 1h ça fait 6*10min)
shift_steps = shift_hours * 6  
#on décale la colonne des résultats (target) de 1h parce que c'est ce qu'on cherche à prédire
dataframe_targets = dataframe[target_names].shift(-shift_steps)

dataframe.head(10)
dataframe_targets.head()
dataframe_targets.tail()
"""
sépraration des données en données d'entré (x) et données de sortie (y) 
tout en retirant les données ayant NAN (not a number) en valeur suite 
au décalage effectué
"""
"""
Attention: la commande dataframe.values permet de ne garder que les données;
on passe donc de la structure dataframe à la structure float64;
c'est très important car on ne peut pas faire certaines opération sur un dataframe
"""
x_data = dataframe.values[0:-shift_steps]
y_data = dataframe_targets.values[:-shift_steps]
#donne le nombre de données d'entré
num_data = len(x_data)
"""
séparation des données en données d'entrainement et de test
On doit le faire pour pouvoir vérifier que notre modèle n'est pas trop 
spécialisé. Donc pour vérifier qu'avec des données inconnues (test), il
peut quand même effectuer un travail correct de prédiction. 
"""
train_split = 0.9
num_train = int(train_split * num_data)

num_test = num_data - num_train

x_train = x_data[0:num_train]
x_test = x_data[num_train:]

y_train = y_data[0:num_train]
y_test = y_data[num_train:]

num_x_signals = x_data.shape[1]

num_y_signals = y_data.shape[1]

"""égalisation des données"""
#on va remettre toutes les données sur une même échelle comprise entre -1 et 1
#car la machine sera plus performante si elle opère avec des données ainsi.
"""Voir GRU, SIGMOID et problème du gradient et réarranger les commentaires"""
#We first create a scaler-object for the input-signals
x_scaler = MinMaxScaler()
#We then detect the range of values from the training-data and scale the training-data.
x_train_scaled = x_scaler.fit_transform(x_train)
#We use the same scaler-object for the input-signals in the test-set.
x_test_scaled = x_scaler.transform(x_test)
#The target-data comes from the same data-set as the input-signals, 
#because it is the weather-data for one of the cities that is merely 
#time-shifted. But the target-data could be from a different source with 
#different value-ranges, so we create a separate scaler-object for the target-data.
y_scaler = MinMaxScaler()
y_train_scaled = y_scaler.fit_transform(y_train)
y_test_scaled = y_scaler.transform(y_test)


"""comme on utilise un fit_batch, il nous faut une fonction générateur de batch"""
def batch_generator(batch_size, sequence_length):
    """
    Generator function for creating random batches of training-data.
    """

    # Infinite loop.
    while True:
        # Allocate a new array for the batch of input-signals.
        x_shape = (batch_size, sequence_length, num_x_signals)
        x_batch = np.zeros(shape=x_shape, dtype=np.float16)

        # Allocate a new array for the batch of output-signals.
        y_shape = (batch_size, sequence_length, num_y_signals)
        y_batch = np.zeros(shape=y_shape, dtype=np.float16)

        # Fill the batch with random sequences of data.
        for i in range(batch_size):
            # Get a random start-index.
            # This points somewhere into the training-data.
            idx = np.random.randint(num_train - sequence_length)
            
            # Copy the sequences of data starting at this index.
            x_batch[i] = x_train_scaled[idx:idx+sequence_length]
            y_batch[i] = y_train_scaled[idx:idx+sequence_length]
        
        yield (x_batch, y_batch)
#donne le nombre d'intervalles qui vont être créés (nombre de tableau de données)
#Plus batch_size est grand, plus il demandera de ressources au GPU. 
#Un grand nombre permet donc d'utiliser au maximum les capacités de la machine.
batch_size = 256
#Donne la taille des intervalles 
#on veut utiliser les données d'une semaine pour prédire la température dans 1h
#Les intervalles auront donc une taille de 6*24*7 = 1008
sequence_length = 6 * 24 * 7

#formation des intervalles
generator = batch_generator(batch_size=batch_size,
                            sequence_length=sequence_length)
x_batch, y_batch = next(generator)


#créé un vecteur à partir de x_batch
#ce vecteur est la première colonne du premier tableau du set de tableaux 
#de x_batch (256, 1008, 3)
#on créé un vecteur de vecteurs (donc on a un tableau de 2 éléments qui sont 
#un tableau de 1973x3 données et un tableau de 1973x1 données)
#preparation des données de validation
validation_data = (np.expand_dims(x_test_scaled, axis=0),
                   np.expand_dims(y_test_scaled, axis=0))


"""Création du modèle"""
"""
Normalement que vaut le input shape?
En entrée on met 256 tableaux de 3 colones et 1008 lignes
mais si l'on met des input de longueur variable avec 3 paramètres différents,
on doit mettre (None, 3,) car num_x_signals = 3
"""
"""expliqué dans le rapport"""
model = Sequential()
model.add(GRU(units=70,
              return_sequences=True,
              input_shape=(None, num_x_signals,)))
#si on ne précise pas, la fonction d'ativation de base est tanh et, 
#la fonction d'activation réccurente est sigmoid
#de base, il y a 512 units et input shape était à 256
model.add(Dense(num_y_signals, activation='linear'))

#La machine n'est pas très forte au tout début donc on a créé une fonction de 
#perte qui commence réellement après les 50 premières itérations. Comme ça,
#elle n'est pas influancée par les très mauvais résultats du début.

warmup_steps = 50

def loss_mse_warmup(y_true, y_pred):

    #Calculate the Mean Squared Error between y_true and y_pred,
    #but ignore the beginning "warmup" part of the sequences.
    
    #y_true is the desired output.
    #y_pred is the model's output.


    # The shape of both input tensors are:
    # [batch_size, sequence_length, num_y_signals].

    # Ignore the "warmup" parts of the sequences
    # by taking slices of the tensors.
    y_true_slice = y_true[:, warmup_steps:, :]
    y_pred_slice = y_pred[:, warmup_steps:, :]

    # These sliced tensors both have this shape:
    # [batch_size, sequence_length - warmup_steps, num_y_signals]

    # Calculate the MSE loss for each value in these tensors.
    # This outputs a 3-rank tensor of the same shape.
    loss = tf.losses.mean_squared_error(labels=y_true_slice,
                                        predictions=y_pred_slice)

    # Keras may reduce this across the first axis (the batch)
    # but the semantics are unclear, so to be sure we use
    # the loss across the entire tensor, we reduce it to a
    # single scalar with the mean function.
    loss_mean = tf.reduce_mean(loss)

    return loss_mean


"""dire ce que fait chaque callback et pourquoi on a choisi tel valeur"""
optimizer = RMSprop(lr=1e-3)
model.compile(loss=loss_mse_warmup, optimizer=optimizer)
model.summary()

#création des fonctions callback
#Créé un checkpoint. 
#Quand une epoch se termine, il sauvegarde temporairement le modèle dans ce 
#checkpoint à partir duquel il effectuera l'entrainement de la prochaine epoch.
#si la nouvelle epoch n'est pas aussi bonne que la précédente, il ne sauvegarde pas.
path_checkpoint = '23_checkpoint.keras'
callback_checkpoint = ModelCheckpoint(filepath=path_checkpoint,
                                      monitor='val_loss',
                                      verbose=1,
                                      save_weights_only=True,
                                      save_best_only=True)

#Fonction qui arrête l'apprentissage si il n'y a pas d'amélioration du modèle 
#après 5 epoch (défini par le paramètre patience = 5)
callback_early_stopping = EarlyStopping(monitor='val_loss',
                                        patience=5, verbose=1)
#permet de visualisé l'évolution de l'apprentissage du modèle
#ici, il est désactivé
callback_tensorboard = TensorBoard(log_dir='./23_logs/',
                                   histogram_freq=0,
                                   write_graph=False)
#Si l'amélioration du modèle n'est pas assez importante, la fonction va
#automatiquement réduire le learning rate d'un facteur 10^-1, avec pour limite
#lr = 10^-4.
callback_reduce_lr = ReduceLROnPlateau(monitor='val_loss',
                                       factor=0.1,
                                       min_lr=1e-4,
                                       patience=0,
                                       verbose=1)

callbacks = [callback_early_stopping,
             callback_checkpoint,
             callback_tensorboard,
             callback_reduce_lr]

"""entrainement"""
# !!!!! Attention, l'opération suivante prend bcp de temps :( !!!!!

model.fit_generator(generator=generator,
                    epochs=20,
                    steps_per_epoch=100,
                    validation_data=validation_data,
                    callbacks=callbacks)
#les variables de bases sont: epochs=20, steps_per_epoch=100, devrait être (1, 50)

#vérifie si il a bien créé le checkpoint
try:
    model.load_weights(path_checkpoint)
except Exception as error:
    print("Error trying to load checkpoint.")
    print(error)

#donne l'erreur finale entre la réalité et la prédiction
result = model.evaluate(x=np.expand_dims(x_test_scaled, axis=0),
                        y=np.expand_dims(y_test_scaled, axis=0))

#affiche la différence entre la vérité et la prédiction
print("loss (test-set):", result)


#fonction pour plotter la comparaison entre la réalité et la prédiction
def plot_comparison(start_idx, length=100, train=True):
    """
    Plot the predicted and true output-signals.
    
    :param start_idx: Start-index for the time-series.
    :param length: Sequence-length to process and plot.
    :param train: Boolean whether to use training- or test-set.
    """
    
    if train:
        # Use training-data.
        x = x_train_scaled
        y_true = y_train
    else:
        # Use test-data.
        x = x_test_scaled
        y_true = y_test
    
    # End-index for the sequences.
    end_idx = start_idx + length
    
    # Select the sequences from the given start-index and
    # of the given length.
    x = x[start_idx:end_idx]
    y_true = y_true[start_idx:end_idx]
    
    # Input-signals for the model.
    x = np.expand_dims(x, axis=0)

    # Use the model to predict the output-signals.
    y_pred = model.predict(x)
    
    # The output of the model is between 0 and 1.
    # Do an inverse map to get it back to the scale
    # of the original data-set.
    y_pred_rescaled = y_scaler.inverse_transform(y_pred[0])
    
    # For each output-signal.
    for signal in range(len(target_names)):
        # Get the output-signal predicted by the model.
        signal_pred = y_pred_rescaled[:, signal]
        
        # Get the true output-signal from the data-set.
        signal_true = y_true[:, signal]

        # Make the plotting-canvas bigger.
        plt.figure(figsize=(15,5))
        
        # Plot and compare the two signals.
        plt.plot(signal_true, label='true')
        plt.plot(signal_pred, label='pred')
        
        # Plot grey box for warmup-period.
        #p = plt.axvspan(0, warmup_steps, facecolor='black', alpha=0.15)
        
        # Plot labels etc.
        plt.ylabel(target_names[signal])
        plt.legend()
        plt.show()

plt.figure()
plot_comparison(start_idx=6000, length=1000, train=True)

plt.figure()
plot_comparison(start_idx=2000, length=1000, train=True)

plt.figure()
plot_comparison(start_idx=200, length=1000, train=False)

plt.figure()
plot_comparison(start_idx= 800, length = 200, train = True)

plt.figure()
plot_comparison(start_idx= 800, length = 200, train = False)




"""
sauvegarde et export du modèle pour une future utilisation dans notre 
application de machine learning sur notre tablette
"""
#Donne les noms des couches d'entré et de sortie (pour le programme Android)
model.input.name[:-2]
model.output.name[:-2]

#Fonction qui va sauvegarder le modèle dans l'état dans lequel il se trouve actuellement.
def freeze_session(session, keep_var_names=None, output_names=None, clear_devices=True):
    """
    Freezes the state of a session into a pruned computation graph.

    Creates a new computation graph where variable nodes are replaced by
    constants taking their current value in the session. The new graph will be
    pruned so subgraphs that are not necessary to compute the requested
    outputs are removed.
    @param session The TensorFlow session to be frozen.
    @param keep_var_names A list of variable names that should not be frozen,
                          or None to freeze all the variables in the graph.
    @param output_names Names of the relevant graph outputs.
    @param clear_devices Remove the device directives from the graph for better portability.
    @return The frozen graph definition.
    """
    graph = session.graph
    with graph.as_default():
        freeze_var_names = list(set(v.op.name for v in tf.global_variables()).difference(keep_var_names or []))
        output_names = output_names or []
        output_names += [v.op.name for v in tf.global_variables()]
        input_graph_def = graph.as_graph_def()
        if clear_devices:
            for node in input_graph_def.node:
                node.device = ""
        frozen_graph = tf.graph_util.convert_variables_to_constants(
            session, input_graph_def, output_names, freeze_var_names)
        return frozen_graph

from keras import backend as K

# Create, compile and train model...

frozen_graph = freeze_session(K.get_session(),
                              output_names=[out.op.name for out in model.outputs])
#On indique le répertoire où sera enregistré le modèle.
tf.train.write_graph(frozen_graph, "D:/Projet BA3/" , "modelFinal.pb", as_text=False)

#Vérification que le fichier à correctement été enregistré
g = tf.GraphDef()
g.ParseFromString(open("D:/Projet BA3/modelFinal.pb","rb+").read())
[n for n in g.node if n.name.find("input") != -1]