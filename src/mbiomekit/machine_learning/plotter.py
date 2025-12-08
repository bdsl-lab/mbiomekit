import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from joblib import load
import shap
import os

plt.rcParams['font.size'] = 7
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['lines.linewidth'] = 0.6


class RandomForestPlotter:
    def __init__(self):
        self.model = None
        self.output_dir = None

    def load_results(self, result_dir):
        # CV models
        self.cv_models = load(f'{result_dir}/cv_models.joblib')
        self.cv_scores_mean = load(f'{result_dir}/cv_scores_mean.joblib')
        self.cv_scores_std = load(f'{result_dir}/cv_scores_std.joblib')
        self.cv_roc_data = load(f'{result_dir}/cv_roc_data.joblib')

        # Final models
        self.model = load(f'{result_dir}/model.joblib')
        self.scores = load(f'{result_dir}/scores.joblib')
        self.roc_data = load(f'{result_dir}/roc_data.joblib')

        # Input data
        input_data = load(f'{result_dir}/input_data.joblib')
        self.feature_table = input_data['feature_table']
        self.labels = input_data['labels']
        self.classes = input_data['classes']
        self.X = input_data['X']
        self.y = input_data['y']

        # SHAP values
        self.shap_values = load(f'{result_dir}/shap_values.joblib')
    
    def set_output_dir(self, dir):
        self.output_dir = dir
        os.makedirs(self.output_dir, exist_ok=True)

    def plot_cv_roc_curve(self, color='blue'):
        # 1. Define a common FPR grid
        mean_fpr = np.linspace(0, 1, 100)

        # 2. Interpolate each fold's TPR
        interp_tprs = []
        for fpr, tpr, thr in self.cv_roc_data:
            interp_tpr = np.interp(mean_fpr, fpr, tpr)
            interp_tpr[0] = 0.0  # ensure it starts at 0
            interp_tprs.append(interp_tpr)

        # 3. Average the interpolated TPRs
        mean_tpr = np.mean(interp_tprs, axis=0)
        mean_tpr[-1] = 1.0  # ensure it ends at 1
        std_tpr = np.std(interp_tprs, axis=0)

        fig = plt.figure(figsize=(3, 3))
        ax = fig.add_subplot(111)

        tpr_upper = np.minimum(mean_tpr + std_tpr, 1)
        tpr_lower = np.maximum(mean_tpr - std_tpr, 0)
        ax.fill_between(mean_fpr, tpr_lower, tpr_upper, color=color, alpha=0.2,
                        label=r'$\pm$ 1 std. dev.', zorder=5)

        ax.plot(mean_fpr, mean_tpr, c='k', linewidth=1, zorder=10)
        ax.plot([0, 1], [0, 1], c='gray',
                linewidth=0.5, linestyle='--', zorder=5)
        ax.grid(zorder=2)

        ax.set_xlabel('False Positive Rate (1-Specificity)')
        ax.set_ylabel('True Positive Rate (Sensitivity)')

        ax.legend()
        
        if self.output_dir:
            fig.savefig(f'{self.output_dir}/roc_curve.pdf')

        print('Average AUC:', self.cv_scores_mean['auc'])

    def plot_feature_importance(self, n_top=20, color='blue'):
        # sns.scatterplot(self.model.feature_importances_)

        # importance
        fi = pd.DataFrame(np.array(
            [self.feature_table.columns, self.model.feature_importances_]).T, columns=['feature', 'importance'])
        fi.sort_values('importance', ascending=False, inplace=True)

        cnt = 0
        top_n = []
        rs = []
        for id, row in fi.iterrows():
            rs.append([row['feature'], row['importance']])
            cnt += 1
            top_n.append(row['feature'])
            if cnt == n_top:
                break
        rs = pd.DataFrame(rs, columns=['feature', 'importance'])
        fig = plt.figure(figsize=(2, 3))
        ax = fig.add_subplot(111)
        sns.barplot(rs, x='importance', y='feature', ax=ax, facecolor=color)
        plt.show()
        
        if self.output_dir:
            fig.savefig(f'{self.output_dir}/feature_importance.pdf')

    def plot_shap_values(self, n_top=20, class_order=None, class_colors=None):
        """
        Select the top 20 influencial features.
        Display their contribution for each classes.
        Display individual contributions by using beeswarm plot
        """
        shap_values = self.shap_values

        class_names = self.classes
        feature_names = self.feature_table.columns
        
        if not class_order:
            class_order = list(class_names)

        n_class = len(class_names)
        shap_values_list = [shap_values[:, :, i] for i in range(n_class)]
        X = self.X

        # 1. Compute mean(|SHAP|) for each class and each feature
        mean_abs_by_class = [np.abs(sv).mean(axis=0) for sv in shap_values_list]  # list of arrays (n_features,)
        mean_abs_matrix = np.vstack(mean_abs_by_class).T  # shape: (n_features, n_classes)

        # 2. Compute global importance and sort
        global_importance = mean_abs_matrix.sum(axis=1)
        sorted_idx = np.argsort(global_importance)[::-1]
        sorted_idx = sorted_idx[:n_top]

        # 3. Slice the data
        mean_abs_matrix_top = mean_abs_matrix[sorted_idx]
        feature_names_top = np.array(feature_names)[sorted_idx]

        # 4. Plot stacked horizontal bar chart
        fig, ax = plt.subplots(figsize=(2, 3))
        bottom = np.zeros(n_top)

        for c in class_order:
            i = class_order.index(c)
            
            if class_colors: 
                ax.barh(range(n_top), mean_abs_matrix_top[:, i], left=bottom, color=class_colors[c], label=f"{c}")
            else:
                ax.barh(range(n_top), mean_abs_matrix_top[:, i], left=bottom, label=f"{c}")
            bottom += mean_abs_matrix_top[:, i]

        ax.set_yticks(range(n_top))
        ax.set_yticklabels(feature_names_top)
        ax.invert_yaxis()
        ax.set_xlabel("mean(|SHAP value|) (average impact on model output)")
        ax.legend()
        plt.show()
        
        if self.output_dir:
            fig.savefig(f'{self.output_dir}/shap_value_multiclasses.pdf')

        for k, sv in enumerate(shap_values_list):
            shap.summary_plot(sv, X,
                              max_display=n_top,
                              feature_names=feature_names,
                              class_names=[class_names[k]],
                              plot_size=1,
                              show=False)
            fig = plt.gcf()
            fig.set_size_inches(2, 3)
            ax = plt.gca()
            ax.tick_params(axis='both', labelsize=7)
            ax.set_xlabel(ax.get_xlabel(), fontsize=7)
            ax.set_ylabel(ax.get_ylabel(), fontsize=7)
            ax.set_title(class_names[k])
            plt.show()
            if self.output_dir:
                fig.savefig(f'{self.output_dir}/shap_c{k}.pdf')

