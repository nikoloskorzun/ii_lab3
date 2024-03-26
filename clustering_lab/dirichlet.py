from sklearn.mixture import BayesianGaussianMixture

def dirichlet_process_mixture(data, k, weight):

    dpmm = BayesianGaussianMixture(n_components=k, covariance_type='full', weight_concentration_prior=weight)
    dpmm.fit(data)
    return dpmm.predict(data), dpmm.predict_proba(data)
