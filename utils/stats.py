import torch
from IPython.display import clear_output
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText
import numpy as np 
import matplotlib.pyplot as plt 
import json 

class BaseStats(object):
    
    def __init__(self, stats=None): 
        self.stats = stats
        
    @classmethod
    def load(cls, input_filename):
        with open(input_filename, 'r') as file:
            return cls(stats=json.load(fp=file))
        
    def save(self, output_filename):
        with open(output_filename, 'w') as file:
            json.dump(fp=file, obj=copy)
    
    @staticmethod
    def compute_batch_stats(batch):
        return torch.max(batch).item(), torch.mean(batch).item(), torch.std(batch).item(), torch.min(batch).item()
    
    @staticmethod
    def add_batch_stats(storage, batch):
        storage.append(BaseStats.compute_batch_stats(batch))
        
    @staticmethod
    def read_batch_stats(storage):
        return tuple(zip(*storage))
    
    @staticmethod
    def last_batch_stats(storage):
        last = lambda x: x[-1]
        return map(last, storage)
    
    @staticmethod
    def plot_batch_stats(axis, storage, title):
        maxs, means, stds, mins = BaseStats.read_batch_stats(storage)
        axis.set_title(title)
        iters = np.arange(len(storage))
        axis.plot(iters, maxs, label='Max')
        axis.errorbar(iters, means, stds, label='Mean')
        axis.plot(iters, mins, label='Min')
        axis.legend(loc='best')
        
    @staticmethod
    def plot_simple_line(axis, values, title, label):
        axis.set_title(title)
        axis.plot(values, label=label)
        axis.legend(loc='best')
        
    @staticmethod
    def plot_probabilities_hist(axis, values, title, label):
        axis.set_title(title)
        axis.hist(values, bins=100, range=(0, 1), label=label)
        axis.legend(loc='best')
        
    @staticmethod
    def plot_description(axis, description):
        text_panel = AnchoredText(description, loc='center')
        axis.add_artist(text_panel)
        
        axis.tick_params(
            axis='both', which='both',
            bottom=False, top=False, left=False, right=False,
            labelbottom=False, labeltop=False, labelleft=False, labelright=False,
        )

        for tl in axis.spines.values():
            tl.set_visible(False)
          
    @staticmethod 
    def subplot_line(grid_size, row_number, max_col_number):
        return (plt.subplot2grid(grid_size, (row_number, i)) for i in range(max_col_number))
    
    def create_figure(self):
        raise NotImplemented("Can't plot anything in base class")
        
    def save_plot(self, output_filename, figure=None):
        if figure is None:
            figure = self.create_figure()
        figure.savefig(output_filename)
    
    def plot(self, output_filename):
        figure = self.create_figure()
        self.save_plot(output_filename, figure)
#         figure.show()
        return figure 
        
    @staticmethod
    def pp_json(json_thing, sort=False, indents=4):
        if type(json_thing) is str:
            return json.dumps(json.loads(json_thing), sort_keys=sort, indent=indents)
        else:
            return json.dumps(json_thing, sort_keys=sort, indent=indents)
        
    def __repr__(self):
        return self.pp_json(self.stats, False)
    
class SRStats(BaseStats):
    '''
    Contains SR generator net stats
    '''
    def __init__(self, stats=None): 
        super(SRStats, self).__init__(stats)
        if stats is None:
            self.stats = dict(
                errors = dict(
                    train=list(),
                    test=list()
                ),
                metrics = dict(
                    train=list(),
                    test=list()        
                )    
            )

        
    def add_set_stats(self, label, batch_errs, batch_metrics):
        self.add_batch_stats(self.stats['errors'][label], batch_errs)
        self.add_batch_stats(self.stats['metrics'][label], batch_metrics)

        
    def last_batch_value_description(self, value='errors'):
        return 'descr'
#         return '''
#         Last batch train max: {:.5f}
#         Last batch train mean, std: {:.5f}, {:.5f}
#         Last batch train min: {:.5f}
        
#         Last batch test max: {:.5f}
#         Last batch test mean, std: {:.5f}, {:.5f}
#         Last batch test min: {:.5f}
#         '''.format(
#             *self.last_batch_stats(self.stats[value]['train']), 
#             *self.last_batch_stats(self.stats[value]['test'])
#         )
    
    
    def create_figure(self, figsize=(16, 6)):
        figure = plt.figure(figsize=figsize)
        grid_size = (2, 3)
        
        train_err_ax, test_err_ax, descr_err_ax = self.subplot_line(grid_size, 0, 3)
        train_metric_ax, test_metric_ax, descr_metric_ax = self.subplot_line(grid_size, 1, 3)
                
        self.plot_batch_stats(train_err_ax, self.stats['errors']['train'], 'Train MSE error')
        self.plot_batch_stats(test_err_ax, self.stats['errors']['test'], 'Test MSE error')
        self.plot_description(descr_err_ax, self.last_batch_value_description('errors'))
        
        self.plot_batch_stats(train_metric_ax, self.stats['metrics']['train'], 'Train PSNR metric')
        self.plot_batch_stats(test_metric_ax, self.stats['metrics']['test'], 'Test PSNR metric')
        self.plot_description(descr_metric_ax, self.last_batch_value_description('metrics'))
        
        plt.tight_layout()
        return figure
    
    
class GANStats(BaseStats):
    def __init__(self, stats=None): 
        super(GANStats, self).__init__(stats)
        if stats is None:
            self.stats = dict(
                generator = dict(
                    feature_loss=dict(
                        train=list(),
                        test=list()
                    ),
                    adversarial_loss=dict(
                        train=list(),
                        test=list()
                    )
                ),
                disciminator = dict(
                    adversarial_loss=dict(
                        train=list(),
                        test=list()
                    ),
                    accuracy=dict(
                        train=list(),
                        test=list()
                    ),
                    predictions = dict(
                        train=list(),
                        test=list()
                    )
                )    
            )

    def add_generator_stats(self, label, feature_loss, adversarial_loss):
        self.add_batch_stats(self.stats['generator']['feature_loss'][label], feature_loss)
        self.add_batch_stats(self.stats['generator']['adversarial_loss'][label], adversarial_loss)

        
    def add_discriminator_stats(self, label, predictions, accuracy, adversarial_loss):
        self.stats['discriminator']['adversarial_loss'][label].append(adversarial_loss)
        self.stats['discriminator']['adversarial_loss'][label].append(accuracy)
        self.stats['discriminator']['predictions'][label].append(list(predictions.numpy().astype(float)))
    
    
    def plot_generator_feature_loss_line(self, train_ax, test_ax, descr_ax):
        self.plot_batch_stats(train_ax, self.stats['generator']['feature_loss']['train'], 'Train G feature loss')
        self.plot_batch_stats(test_ax, self.stats['generator']['feature_loss']['test'], 'Test G feature loss')
        self.plot_description(descr_ax, 'G feature loss description')
        
        
    def plot_model_adv_loss_line(self, train_ax, test_ax, descr_ax, model_type):
        train_ax_title = 'Train {} adversarial loss'.format(model_type)
        self.plot_simple_line(train_ax, self.stats[model_type]['adversarial_loss']['train'], train_ax_title, 'loss')
        test_ax_title = 'Test {} adversarial loss'.format(model_type)
        self.plot_simple_line(test_ax, self.stats[model_type]['adversarial_loss']['test'], test_ax_title, 'loss')
        self.plot_description(descr_ax, '{} adversarial loss description'.format(model_type))
   

    def plot_discriminator_accuracy_line(self, train_ax, test_ax, descr_ax):
        train_ax_title = 'Train discriminator accuracy'
        self.plot_simple_line(train_ax, self.stats['discriminator']['accuracy']['train'], train_ax_title, 'accuracy')
        test_ax_title = 'Test discriminator accuracy'
        self.plot_simple_line(test_ax, self.stats['discriminator']['accuracy']['test'], test_ax_title, 'accuracy')
        self.plot_description(descr_ax, 'Discriminator accuracy description')

        
    def plot_discriminator_predictions_line(self, train_ax, test_ax, descr_ax):
        values = self.stats['discriminator']['predictions']['train'][-1]
        self.plot_probabilities_hist(train_ax, values, 'Train discriminator predictions', 'Predictions')
        values = self.stats['discriminator']['predictions']['test'][-1]
        self.plot_probabilities_hist(test_ax, values, 'Test discriminator predictions', 'Predictions')
        self.plot_description(descr_ax, 'Discriminator predictcions description')
        
    
    def create_figure(self, figsize=(20, 20)):
        figure = plt.figure(figsize=figsize)
        grid_size = (5, 3)
        
        train_feature_ax, test_feature_ax, descr_feature_ax = self.subplot_line(grid_size, 0, 3)
        self.plot_generator_feature_loss_line(train_feature_ax, test_feature_ax, descr_feature_ax)
        
        train_gen_adv_ax, test_gen_adv_ax, descr_gen_adv_ax = self.subplot_line(grid_size, 1, 3)
        self.plot_model_adv_loss_line(train_gen_adv_ax, test_gen_adv_ax, descr_gen_adv_ax, 'generator')
        
        train_predictions_ax, test_predictions_ax, descr_predictions_ax = self.subplot_line(grid_size, 2, 3)
        self.plot_discriminator_accuracy_line(train_predictions_ax, test_predictions_ax, descr_predictions_ax)
        
        train_discr_acc_ax, test_discr_acc_ax, descr_discr_acc_ax = self.subplot_line(grid_size, 3, 3)
        self.plot_discriminator_predictions_line(train_discr_acc_ax, test_discr_acc_ax, descr_discr_acc_ax)
        
        plt.tight_layout()
        return figure
    