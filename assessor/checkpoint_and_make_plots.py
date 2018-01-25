import os
if 'voleti.vikram' in os.getcwd():
    import matplotlib
    matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np

from keras.callbacks import Callback

#########################################################
# CALLBACK
#########################################################


class CheckpointAndMakePlots(Callback):

    # Init
    def __init__(self, file_name_pre="assessor_cnn_adam", save_dir="."):
        self.file_name_pre = file_name_pre
        self.save_dir = save_dir

    # On train start
    def on_train_begin(self, logs={}):
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
        self.best_val_loss = 1000

    # At every epoch
    def on_epoch_end(self, epoch, logs={}):

        # Get
        tl = logs.get('loss')
        ta = logs.get('acc')
        vl = logs.get('val_loss')
        va = logs.get('val_acc')
        print("\n", tl, ta, vl, va)

        # Append losses and accs
        self.train_losses.append(tl)
        self.val_losses.append(vl)
        self.train_accuracies.append(ta)
        self.val_accuracies.append(va)

        # Save model
        if vl < self.best_val_loss:
            self.best_val_loss = vl
            self.save_model_checkpoint(epoch, tl, ta, vl, va)

        # Save history
        self.save_history()

        # Plot graphs
        self.plot_and_save_losses_and_accuracies(epoch)

    # Save model checkpoint
    def save_model_checkpoint(self, epoch, tl, ta, vl, va):
        model_file_path = os.path.join(self.save_dir,
            self.file_name_pre + "_epoch{0:03d}_tl{1:.4f}_ta{2:.4f}_vl{3:.4f}_va{4:.4f}.hdf5".format(epoch, tl, ta, vl, va))
        print("Saving model", model_file_path)
        self.model.save_weights(model_file_path)

    def save_history(self):
        print("Saving history in", self.save_dir)
        np.savetxt(os.path.join(self.save_dir, self.file_name_pre + "_loss_history.txt"), self.train_losses, delimiter=",")
        np.savetxt(os.path.join(self.save_dir, self.file_name_pre + "_acc_history.txt"), self.train_accuracies, delimiter=",")
        np.savetxt(os.path.join(self.save_dir, self.file_name_pre + "_val_loss_history.txt"), self.val_losses, delimiter=",")
        np.savetxt(os.path.join(self.save_dir, self.file_name_pre + "_val_acc_history.txt"), self.val_accuracies, delimiter=",")

    # Plot and save losses and accuracies
    def plot_and_save_losses_and_accuracies(self, epoch):
        print("Saving plot for epoch", str(epoch), ":",
            os.path.join(self.save_dir, self.file_name_pre + "_plots.png"))

        plt.subplot(121)
        plt.plot(self.train_losses, label='train_loss')
        plt.plot(self.val_losses, label='val_loss')
        leg = plt.legend(loc='upper right', fontsize=11, fancybox=True)
        leg.get_frame().set_alpha(0.3)
        plt.xlabel('epochs')
        plt.ylabel('loss')
        plt.title("Loss")

        plt.subplot(122)
        plt.plot(self.train_accuracies, label='train_acc')
        plt.plot(self.val_accuracies, label='val_acc')
        leg = plt.legend(loc='lower right', fontsize=11, fancybox=True)
        leg.get_frame().set_alpha(0.3)
        plt.xlabel('epochs')
        plt.ylabel('acc')
        plt.yticks(np.arange(0, 1.05, 0.05))
        plt.tick_params(axis='y', which='both',
                        labelleft='on', labelright='on')
        plt.gca().yaxis.grid(True)
        plt.title("Accuracy")

        plt.tight_layout()
        # plt.subplots_adjust(top=0.85)
        plt.suptitle(self.file_name_pre, fontsize=10)
        plt.savefig(os.path.join(self.save_dir,
                                 self.file_name_pre + "_plots_loss_acc.png"))
        plt.close()


