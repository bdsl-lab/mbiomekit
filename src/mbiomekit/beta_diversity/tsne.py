import pandas as pd 
from sklearn.manifold import TSNE
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from sklearn.datasets import make_blobs


plt.rcParams['font.size'] = 7
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['lines.linewidth'] = 0.6      # affects whiskers, medians, caps (all Line2D)


class TSNEAnaysis:
    def __init__(self):
        self.dmat = None
        self.tsne_crds = None
        self.tsne_crds_cluster = None
        
        self.aic_bic = None
        
    def load_distance_matrix(self, dmat):
        self.dmat = dmat
    
    def tsne(self):
        tf = TSNE(n_components=2, learning_rate='auto', init='random', perplexity=20, metric="precomputed", random_state=1234).fit_transform(self.dmat)
        tf = pd.DataFrame(tf, columns=['tSNE-1', 'tSNE-2'])
        tf.index = self.dmat.index
        self.tsne_crds = tf 
    
    def choose_cluster_model(self, fout='tsne_aic_bic.tsv'):
        # data 
        X = self.tsne_crds.loc[:, ['tSNE-1', 'tSNE-2']].values

        # Range for number of clusters
        ns = range(1, 11)

        # Lists to store AIC and BIC values
        aic_values = []
        bic_values = []

        # Fit GMM for each number of clusters and compute AIC and BIC
        for n in ns:
            gmm = GaussianMixture(n_components=n, random_state=1234)
            gmm.fit(X)
            aic_values.append(gmm.aic(X))
            bic_values.append(gmm.bic(X))
            
        self.aic_bic = pd.DataFrame({
            'Number of clusters': ns, 
            'AIC': aic_values, 
            'BIC': bic_values
        })
        self.aic_bic.set_index('Number of clusters', inplace=True)
        self.aic_bic.to_csv(fout, sep='\t')
        
    def cluster(self, n_clusters=4, fout='tsne_cluster.tsv'):
        # data 
        X = self.tsne_crds.loc[:, ['tSNE-1', 'tSNE-2']].values

        # cluster 
        gmm = GaussianMixture(n_components=n_clusters, random_state=1234)
        gmm.fit(X)
        clusters = gmm.predict(X)
        
        self.tsne_crds_cluster = self.tsne_crds.loc[:]
        self.tsne_crds_cluster['cluster'] = clusters
        
        self.tsne_crds_cluster.to_csv(fout, sep='\t')
    
    
class TSNEPlotter:
    def __init__(self):
        pass
    
    def plot_aic_bic(self, fin, fout= 'aic_bic.pdf', palette={'aic': '#9fbcca', 'bic': '#326a81'}):
        aic_bic = pd.read_csv(fin, sep='\t', header=0, index_col=0)
        fig = plt.figure(figsize=(2, 1.25))
        ax = fig.add_subplot(111)
 
        ns = aic_bic.index
        aic_values = aic_bic['AIC']
        bic_values = aic_bic['BIC']
 
        ax.plot(ns, aic_values, label='AIC', marker='o', linewidth=1, c=palette['aic'], zorder=5, markersize=5)
        ax.plot(ns, bic_values, label='BIC', marker='o', linewidth=1, c=palette['bic'], zorder=10, markersize=5)
        ax.set_xlabel('Number of clusters')
        ax.set_ylabel('AIC/BIC')
        ax.set_title('AIC and BIC for Different Numbers of Clusters')
        ax.legend()
        ax.grid(zorder=1)
        
        if fout:
            fig.savefig(fout)
    
    def plot_tsne_cluster(self, fin, fout=None, palette=None):
        tsne_c = pd.read_csv(fin, sep='\t', header=0, index_col=0)

        fig = plt.figure(figsize=(2, 2))
        ax = fig.add_subplot(111)
        
        sns.scatterplot(tsne_c, x='tSNE-1', y='tSNE-2', hue='cluster', ax=ax, zorder=10, palette=palette, s=20)
        ax.grid(zorder=2)

        if fout:
            fig.savefig(fout)
    
    