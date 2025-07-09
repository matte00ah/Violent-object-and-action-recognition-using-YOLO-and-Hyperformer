import os
import matplotlib.pyplot as plt
import json
import numpy as np
import pandas as pd

root_path = '/homes/mbulgarelli/Hyperformer'
file_names = [f for f in os.listdir(root_path) if f.endswith('.json') and f.startswith('NetworkResults-')]
file_names = ['NetworkResults-2601371.json']

for name in file_names:
    f = open(os.path.join(root_path, name))
    data = json.load(f)

    evaluation = []
    train = []
    best = data['Summary']
    setup = data["Setup"].copy()
    for k in ["Dataset", "Model"]:
        if k in setup and isinstance(setup[k], str):
            setup[k] = setup[k].split("/")[-1]
    caption = str(setup)
    results = data['Results']
    for res in results:
        if 'Evaluation Epoch' in res:
            evaluation.append(res)
        elif 'Training Epoch' in res:
            train.append(res)

    # Estrai epoca, accuratezza e loss dalla lista evaluation
    epochs = [e['Evaluation Epoch'] for e in evaluation]
    epochs[-1] = epochs[-2]+1
    accuracies = [e['Accuracy'] for e in evaluation]
    losses = [e['Loss'] for e in evaluation]

    epochs_tr = [e['Training Epoch'] for e in train]
    epochs_tr[-1] = epochs_tr[-2]+1
    accuracies_tr = [e['Accuracy'] for e in train]
    losses_tr = [e['Loss'] for e in train]

    # Crea i dataframe
    df_accuracy = pd.DataFrame({'epoch': epochs, 'Eval_Accuracy': accuracies})
    # df_accuracy['Smoothed_eval'] = df_accuracy['Eval_Accuracy'].rolling(window=10, min_periods=5).mean()
    df_loss = pd.DataFrame({'epoch': epochs, 'Eval_Loss': losses})

    df_accuracy_tr = pd.DataFrame({'epoch': epochs_tr, 'Train_Accuracy': accuracies_tr})
    # df_accuracy_tr['Smoothed_train'] = df_accuracy_tr['Train_Accuracy'].rolling(window=10, min_periods=5).mean()
    df_loss_tr = pd.DataFrame({'epoch': epochs_tr, 'Train_Loss': losses_tr})

    # Trova tutte le righe corrispondenti all'epoca migliore
    best_epoch = best['Best Epoch']
    best_accuracy = best['Best Accuracy']

    # Trova gli indici dove sia epoca, accuratezza e loss coincidono
    best_indices = [
        i for i, (e, a) in enumerate(zip(epochs, accuracies))
        if e == best_epoch and a == best_accuracy 
    ]

    df_accuracy_combined = pd.merge(df_accuracy, df_accuracy_tr, on='epoch')
    # df_accuracy_smoothed_combined = pd.merge(df_accuracy['Smoothed_eval'], df_accuracy_tr['Smoothed_train'], on='epoch')
    df_loss_combined = pd.merge(df_loss, df_loss_tr, on='epoch')

    name = name[:-5]
    # Grafico accuratezza
    ax_acc = df_accuracy_combined.plot(x='epoch', y=['Train_Accuracy', 'Eval_Accuracy'],
                                    # figsize=(10, 6),
                                    title='Accuratezza del Modello durante il Training',
                                    ylabel='Accuratezza',
                                    grid=True,
                                    color={'Train_Accuracy': 'green', 'Eval_Accuracy': 'blue'},
                                    # style={'Eval_Accuracy': '--'} # Style per la linea di valutazione
    ) 
    for idx in best_indices:
        plt.plot(epochs[idx], accuracies[idx], 'ro', label='Best' if idx == best_indices[0] else "")
    if best_indices:
        handles, labels = ax_acc.get_legend_handles_labels()
        if 'Best' not in labels:
            handles.append(plt.Line2D([], [], color='black', marker='o', linestyle='', label='Best', markerfacecolor='black'))
            labels.append('Best')
        ax_acc.legend(handles, labels)
    plt.figtext(0.5, -0.08, caption, wrap=True, horizontalalignment='center', fontsize=10)
    plt.savefig(os.path.join(root_path, f"plots/{name}_Accuracy.png"), bbox_inches='tight')
    plt.close()

    # Grafico loss
    ax_loss = df_loss_combined.plot(x='epoch', y=['Train_Loss', 'Eval_Loss'],
                                #  figsize=(10, 6),
                                 title='Loss del Modello durante il Training',
                                 ylabel='Loss',
                                 grid=True,
                                 color={'Train_Loss': 'green', 'Eval_Loss': 'blue'},
                                #  style={'Eval_Loss': '--'}) # Style per la linea di valutazione
    )
    for idx in best_indices:
        plt.plot(epochs[idx], losses[idx], 'ro', label='Best' if idx == best_indices[0] else "")
    if best_indices:
        handles, labels = ax_loss.get_legend_handles_labels()
        if 'Best' not in labels:
            handles.append(plt.Line2D([], [], color='black', marker='o', linestyle='', label='Best', markerfacecolor='black'))
            labels.append('Best')
        ax_loss.legend(handles, labels)
    plt.figtext(0.5, -0.08, caption, wrap=True, horizontalalignment='center', fontsize=10)
    plt.savefig(os.path.join(root_path, f"plots/{name}_Loss.png"), bbox_inches='tight')
    plt.close()
    print('ciao')