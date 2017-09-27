import numpy as np
from scipy.stats import multivariate_normal
from scipy.spatial.distance import cdist
from sklearn.linear_model.base import BaseEstimator
from scipy.misc import logsumexp
import numpy as np
from sklearn.cluster import KMeans

def compute_labels(X, mu):
    distances = cdist(X, mu)
    labels = np.argmin(distances, axis=1)
    return labels


def log_likelihood(X, w, mu, sigma):
    n_objects, n_features = X.shape
    n_clusters = w.size
    log_gamma = np.zeros((n_objects, n_clusters))
    for cluster in xrange(n_clusters):
        log_gamma[:, cluster] = np.log(w[cluster])
        log_gamma[:, cluster] += multivariate_normal.logpdf(X, mu[cluster, :],
                                                            sigma[cluster, :, :])
    return np.sum(logsumexp(log_gamma, axis=1))



class GMM(BaseEstimator):
    def __init__(self, n_clusters, max_iter=300, n_init=10, min_covar=1e-3, tol=1e-4, init_kmeans=False, logging=False):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.n_init = n_init
        self.init_kmeans = init_kmeans
        self.min_covar = min_covar
        self.tol = tol
        self.logging = logging

    def fit(self, X):

        if self.logging:
            logs = {}

        n_objects = X.shape[0]
        n_features = X.shape[1]

        best_log_likelihood = -np.inf
        
        if self.init_kmeans:
            n_init = 1
        else:
            n_init = self.n_init
            
        for i in xrange(n_init):
            if self.logging:
                logs['log_likelihood'] = []
                logs['mu'] = []
                logs['sigma'] = []
                logs['labels'] = []
                
            if self.init_kmeans:
                kmeans = KMeans(n_clusters=self.n_clusters, max_iter=self.max_iter, tol=self.tol, n_init=self.n_init)
                kmeans.fit(X)
                labels = kmeans.labels_
            else:
                centers_idx = np.random.choice(n_objects, size=self.n_clusters, replace=False)
                mu = X[centers_idx, :]
                labels = compute_labels(X, mu)
                
            clusters_arr, counts = np.unique(labels, return_counts=True)
            w = counts.astype(float) / np.sum(counts)
            mu = np.zeros((self.n_clusters, n_features))
            sigma = np.zeros((self.n_clusters, n_features, n_features))
            
            
            for cluster_idx, cluster in enumerate(clusters_arr):
                idx = (labels == cluster)
                mu[cluster_idx, :] = np.mean(X[idx, :], axis=0)
                if idx.sum() > 1:
                    sigma[cluster_idx, :, :] = np.cov(X[idx, :].T) + np.identity(n_features).astype(np.float) * self.min_covar
                else:
                    sigma[cluster_idx, :, :] = np.identity(n_features).astype(np.float) * self.min_covar

            log_gamma = np.zeros((n_objects, self.n_clusters))
            
            ll = log_likelihood(X, w, mu, sigma)
            converged = False
            it = 0

            while it < self.max_iter and not converged:

                # E-step
                for cluster in xrange(self.n_clusters):
                    log_gamma[:, cluster] = np.log(w[cluster])
                    log_gamma[:, cluster] += multivariate_normal.logpdf(X, mu[cluster],
                                                            sigma[cluster])
            
                log_gamma -= logsumexp(log_gamma, axis=1, keepdims=True)  
                gamma = np.exp(log_gamma)
                n = gamma.sum(axis=0)
                
                # M-step
                for cluster in xrange(self.n_clusters):
                    Y = X.copy()
                    for obj in xrange(n_objects):
                        Y[obj] *= gamma[obj, cluster]
                    mu[cluster] = Y.sum(axis=0)
                    mu[cluster] /= n[cluster]
                    w[cluster] = n[cluster] / n_objects
                    sigma[cluster] = Y.T.dot(X) + np.identity(n_features).astype(np.float) * self.min_covar
                    sigma[cluster] /= n[cluster]
                    sigma[cluster] -= np.outer(mu[cluster], mu[cluster])
                    
                new_ll = log_likelihood(X, w, mu, sigma)
                
                if self.logging:
                    logs['log_likelihood'].append(new_ll)
                    logs['mu'].append(mu)
                    logs['sigma'].append(sigma)
                    logs['labels'].append(np.argmax(gamma, axis=1))
                    
                if new_ll - ll < self.tol:
                    converged = True
                    
                ll = new_ll
                it += 1

            current_log_likelihood = log_likelihood(X, w, mu, sigma)
            if current_log_likelihood > best_log_likelihood:
                self.labels_ = np.argmax(gamma, axis=1)
                self.w_ = w
                self.cluster_centers_ = mu
                self.covars_ = sigma
                if self.logging:
                    self.logs = logs
                best_log_likelihood = current_log_likelihood
                