import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error
from scipy.spatial.distance import cdist
import time
from catboost import CatBoostRegressor
from EarlyStop import CustomEarlyStopCallback
import sys

# Considers the repeated samples
class ActiveLearner():
    def __init__(self, Q, K, n_queries, initial_size, quantiles, batch_size, device, key, B):
        self.n_queries = n_queries
        self.Q = Q
        self.K = K
        self.initial_size = initial_size
        self.quantiles = quantiles
        self.batch_size = batch_size
        self.device = device
        self.key = key
        self.B = B
        
    
    # A single AL experiment:
    def al_iterations(self, data_loader, X_train, X_test, y_train, y_test, seed_initial, seed_rf, evaluator, calculator, seed_QBC_model, seed_QBC_bootstrap):
        
        # For QBC member list
        member_list = []
        
        # Systems list with label list: save the data features and label by system in the training pool
        sys_list = []
        label_list = []
        
        # unlabeled pool: samples grouped by system in the training data, indices_list saves the all indices of samples in the X_train
        sys_list, label_list, indices_list = data_loader.append_systems(sys_list, label_list, X_train, y_train)
        
        # total number of systems
        total_system = len(sys_list)
        print("Total number of system:", total_system)  

        if type(data_loader).__name__ == "SemiConductor":
            X_test = X_test.drop("components", axis='columns')
            union_set_X = union_set_X.drop("components", axis='columns')
    
        # The feature dimensions:
        X = X_test.shape[1]

        # Index of the unlabelled pool by systems:
        X_index_list = []
        for s in np.arange(len(sys_list)):
            # Covert the dataframe of each system data into numpy array:
            sys = sys_list[s].to_numpy(dtype="float64")
            sys_list[s] = sys
            # The indices of samples in each system, which is used for recording the unlabelled pools, saved in X_index_list, ordered by the system list
            X_index_list.append(np.arange(sys.shape[0]))
            
        # Used data: for saving the labelled data
        used_data = np.empty(shape=(0, X))
        used_label = np.empty(shape=(0)).reshape(-1, 1)
        
        # Initialization-randomly select a batch:
        # Array for saving the inital labelled samples
        X_initial = np.empty(shape=(0, X))
        y_initial = np.empty(shape=(0)).reshape(-1, 1)

        # Initialization---Randomly select one system:
        np.random.seed(seed_initial)
        idx = np.random.choice(range(total_system), size=1, replace=False)[0]
        
        selected_system = sys_list[idx]
        selected_system_label = label_list[idx]
        
        # The size of the randomly selected batch:
        random_batch_s = selected_system.shape[0]
        
        # If the selected random batch has not enough samples for initilaization, or in superconductivity dataset, there are some different samples but have the same features: reinitialize the above process:
        while random_batch_s < self.initial_size or (np.unique(selected_system, axis=0).shape[0] == 1):
            idx = np.random.choice(range(total_system), size=1, replace=False)[0]
            # Recorded the selected batch:
            selected_system = sys_list[idx]
            selected_system_label = label_list[idx]
            random_batch_s = selected_system.shape[0]
            
        # 2 - Initialization-randomly select some samples in the selected batch:
        np.random.seed(seed_initial)
        index = np.random.choice(range(selected_system.shape[0]), size=self.initial_size, replace=False)
        # Save the inital samples
        X_initial = selected_system[index]
        y_initial = selected_system_label[index].reshape(-1, 1)

        while (np.unique(X_initial, axis=0).shape[0] == 1):
            index = np.random.choice(range(selected_system.shape[0]), size=self.initial_size, replace=False)
             # Save the inital samples
            X_initial = selected_system[index]
            y_initial = selected_system_label[index].reshape(-1, 1)
            
        # Update the selected labelled samples:
        used_data = np.append(used_data, X_initial, axis=0)
        used_label = np.append(used_label, y_initial, axis=0).reshape(-1, 1)
        
        # Update unlabelled pool indices: idx is the batch index, index is the sample index within the batch
        X_index_list[idx] = np.delete(X_index_list[idx], index, axis=0)
        
        # Learner: multi-quantile predictions
        model, prediction_ = calculator.quantile_regression(used_data, used_label, seed_rf, X_test, X)
        
        # Initialize the 5 committee members for Query-by-Committee method:
        if self.key == "QBC":
            key_early_stop = "RMSE"
            np.random.seed(seed_QBC_bootstrap[0])
            # For each member:
            for m in np.arange(5):
                # Bootstrap the indices with replacement from the labelled samples
                jdx = np.random.choice(np.arange(used_data.shape[0]), used_data.shape[0], replace=True)
                while np.unique(used_data[jdx], axis=0).shape[0] == 1:
                    jdx = np.random.choice(np.arange(used_data.shape[0]), used_data.shape[0], replace=True)
                print(seed_QBC_model[m])
                cbr = CatBoostRegressor(iterations=1000, verbose=0, random_state = seed_QBC_model[m])
                cbr.fit(used_data[jdx], used_label[jdx], callbacks=[CustomEarlyStopCallback(key_early_stop)])
                member_list.append(cbr)

        # Model performance after initialization:
        # For the whole test set
        prediction_point = evaluator.evaluation_metrics(prediction_, X_test, y_test, calculator.get_predictive_distribution, calculator.get_interval_ranges)
        
        testing_r2 = [r2_score(y_test, prediction_point)]
        testing_mse = [mean_squared_error(y_test, prediction_point)]

        print("Initialization R2:", testing_r2)
        print("Initialization MSE:", testing_mse)

        total_time = []
        # AL Process:
        for idx in range(self.n_queries):
            start = time.process_time()
            print('Query no. %d' % (idx+1))
            
            count_batch = 0
            for t in range(len(X_index_list)):
                if X_index_list[t].shape[0] >= self.batch_size:
                    count_batch += 1
            if count_batch < self.B:
                continue
            
            if self.key == "Random":  # Random Sampling Baseline
                # Random Selection of B batch: batch indices
                top_indices = np.random.choice(range(len(X_index_list)), size=self.B, replace=False)
                # To save the selected batch index
                top_indices_list = []
                # To save the sample indices of the batch
                top_query_index_list = []
                
                for top_index in top_indices:
                    top_index_ = top_index
                    while (X_index_list[top_index_].shape[0] < self.batch_size):
                        top_index_new = np.random.choice(range(len(X_index_list)), size=1, replace=False)[0]
                        if (top_indices_list==top_index_new).any() or (top_indices==top_index_new).any():
                            continue
                        else: top_index_ = top_index_new
                    top_indices_list.append(top_index_)
                    # Random selection in the selected system
                    top_query_index = np.random.choice(range(X_index_list[top_index_].shape[0]), size=self.batch_size, replace=False)
                    top_query_index_list.append(top_query_index)
                print(top_indices_list)
                assert np.unique(np.asarray(top_indices_list)).shape[0] == self.B, "The selected Q data batches should be different."
                         
            else:  # The other AL methods
                top_sys = 0
                top_index = 0
                # top_query_index = np.empty(shape=(0, self.batch_size))
                # For saving the top-K acquisition scores:
                top_all_utility_all = []
                # For saving the mean of the top-K acquisition score
                top_all_utility_avg = []
                # batch indices
                top_indices_list = []
                # sample indices within the batch
                top_query_index_list = []
                
                # Each System:
                for sys_idx, sys in enumerate(sys_list):
                    each_system = sys[X_index_list[sys_idx]]
                    if X_index_list[sys_idx].shape[0] >= self.batch_size:
                        # The unlabeled sample indices within each batch
                        unlabeled_index = X_index_list[sys_idx]
                        
                        if self.key == "Greedy-iGS":
                            query_idx, top_k = self.custom_query_strategy_greedy(model, sys[unlabeled_index], used_data, used_label, self.batch_size)
                        elif self.key == "Greedy-GSy":
                            query_idx, top_k = self.custom_query_strategy_greedy_y(model, sys[unlabeled_index], used_data, used_label, self.batch_size)
                        elif self.key == "Greedy-GSx":
                            query_idx, top_k = self.custom_query_strategy_greedy_GSx(sys[unlabeled_index], unlabeled_index, used_data, used_label, self.batch_size)
                        elif self.key == 'QBC':
                            query_idx, top_k = self.query_QBC(sys[unlabeled_index], member_list, self.batch_size)
                        else:
                            assert self.key in ["Greedy-iGS", "Greedy-GSx", "QBC", "Random", "Greedy-GSy"], 'The comparisions methods keywords can only be "Random", "Greedy-iGS", "Greedy-GSx", "Greedy-GSy" or "QBC".'
                        
                        # These for lists have the same dimension
                        top_all_utility_all.append(top_k)
                        top_all_utility_avg.append(top_k.mean())
                        top_indices_list.append(sys_idx)
                        top_query_index_list.append(query_idx)
                        
                # Select the number of B batches that have the largest averaged acquisition scores
                max_sys_idx = self.multi_argmax(np.asarray(top_all_utility_avg), self.B)
                assert np.unique(np.asarray(max_sys_idx)).shape[0] == self.B, "The selected Q data batches should be different."

            end = time.process_time()
            total_time.append(end-start)
            # print("Time:", end-start)
            
            if self.key == "Random":
                for i in range(len(top_indices_list)):
                    # The selected batch
                    top_index = top_indices_list[i]
                    # The selected samples in that batch
                    top_query_index = top_query_index_list[i]
                    
                    sys_index = X_index_list[top_index][top_query_index]
                    new_X = sys_list[top_index][sys_index]
                    top_labels = label_list[top_index]
                    new_y = top_labels[X_index_list[top_index][top_query_index]].reshape(-1, 1).ravel()
                    # Update
                    X_index_list[top_index] = np.delete(X_index_list[top_index], top_query_index, axis=0)
        
                    # Adding the used data to the used_data pool
                    used_data = np.append(used_data, new_X, axis=0)
                    used_label = np.append(used_label, new_y.reshape(-1, 1), axis=0).reshape(-1, 1)
                    print("Queried samples:", np.unique(used_data, axis=0).shape[0])
            
            # For these two methods, return the indices of the selected samples within the batch
            if self.key == "Greedy-iGS" or self.key == "Greedy-GSy" or self.key == "QBC":
                for id_s in max_sys_idx:
                    # Selected sys_list index
                    top_index = top_indices_list[id_s]
                    # Selected sample indices of this batch
                    top_query_index = top_query_index_list[id_s]
                    # The queried sample indices in the system:
                    sys_index = X_index_list[top_index][top_query_index]
                    print("Remaining Samples", X_index_list[top_index].shape)
                    new_X = sys_list[top_index][sys_index]
                    top_labels = label_list[top_index]
                    new_y = top_labels[X_index_list[top_index][top_query_index]].reshape(-1, 1).ravel()
                    # Update
                    X_index_list[top_index] = np.delete(X_index_list[top_index], top_query_index, axis=0)
        
                    # Adding the used data to the used_data pool
                    used_data = np.append(used_data, new_X, axis=0)
                    used_label = np.append(used_label, new_y.reshape(-1, 1), axis=0).reshape(-1, 1)
                    print("Queried samples:", np.unique(used_data, axis=0).shape[0])

            # this method returns the indices of the samples in each original batch, rather than the unlabelled batch   
            if self.key == "Greedy-GSx":
                for id_s in max_sys_idx:
                    top_index = top_indices_list[id_s]
                    top_query_index = top_query_index_list[id_s]
                    
                    new_X = sys_list[top_index][top_query_index]
                    top_labels = label_list[top_index]
                    new_y = top_labels[top_query_index].reshape(-1, 1).ravel()
                    # Update
                    X_index_list[top_index] = np.setdiff1d(X_index_list[top_index], top_query_index, False)
                    # Adding the used data to the used_data pool
                    used_data = np.append(used_data, new_X, axis=0)
                    used_label = np.append(used_label, new_y.reshape(-1, 1), axis=0).reshape(-1, 1)
                    print("Queried samples:", np.unique(used_data, axis=0).shape[0])

            # Model performance after each query:
            # Retrain the QBC Committee:
            if self.key == "QBC":
                member_list = self.QBC_retrain(idx, seed_QBC_bootstrap, member_list, used_data, used_label)
            
            # Model performance after each query:
            model, prediction_ = calculator.quantile_regression(used_data, used_label, seed_rf, X_test, X)
            prediction_point = evaluator.evaluation_metrics(prediction_, X_test, y_test, calculator.get_predictive_distribution, calculator.get_interval_ranges)

            testing_r2.append(r2_score(y_test, prediction_point))
            testing_mse.append(mean_squared_error(y_test, prediction_point))

            print("R2:", r2_score(y_test, prediction_point))
            print("MSE:", mean_squared_error(y_test, prediction_point))
            
        testing_r2 = np.array(testing_r2)
        testing_mse = np.array(testing_mse)
        
        return testing_r2, testing_mse, used_data, used_label

    
    def get_closest_dist(self, unlabeled_data, labeled_data):
        # Should be global info instead of within the clusters
        d = cdist(unlabeled_data, labeled_data, 'euclidean')
        d_x_min = d.min(axis=1)
        return d_x_min

    
    def get_closest_dist_xy(self, unlabeled_predictions, labeled_data_label, unlabeled_data, labeled_data):
        d_y = cdist(np.array(unlabeled_predictions).reshape(-1, 1), labeled_data_label, 'euclidean')
        d_x = cdist(unlabeled_data, labeled_data, 'euclidean')
        d_xy = np.multiply(d_x, d_y)
        d_xy_min = d_xy.min(axis=1)
        return d_xy_min
        

    def get_closest_dist_y(self, unlabeled_predictions, labeled_data_label, unlabeled_data, labeled_data):
        d_y = cdist(np.array(unlabeled_predictions).reshape(-1, 1), labeled_data_label, 'euclidean')
        d_y_min = d_y.min(axis=1)
        return d_y_min

    
    def custom_query_strategy_greedy(self, model, X_pool, used_data, used_label, n_instances):
        prediction_ = model.predict(X_pool)
        q_idx = self.quantiles.index(0.5)
        prediction_ = prediction_[:, q_idx]
        d_xy = self.get_closest_dist_xy(prediction_, used_label, X_pool, used_data)
        utility = d_xy
        selected_indices = self.multi_argmax(utility, n_instances=n_instances)
        top_k = utility[selected_indices]
        return selected_indices, top_k

    
    def custom_query_strategy_greedy_y(self, model, X_pool, used_data, used_label, n_instances):
        prediction_ = model.predict(X_pool)
        q_idx = self.quantiles.index(0.5)
        prediction_ = prediction_[:, q_idx]
        d_y = self.get_closest_dist_y(prediction_, used_label, X_pool, used_data)
        utility = d_y
        selected_indices = self.multi_argmax(utility, n_instances=n_instances)
        top_k = utility[selected_indices]
        return selected_indices, top_k
    
    
    def custom_query_strategy_greedy_GSx(self, X_pool, unlabeled_index, used_data, used_label, n_instances):

        selected_indices = []
        top_k = []

        dist_ul = cdist(X_pool, used_data, metric='euclidean') 
        min_dist = dist_ul.min(axis=1)                         
        for _ in range(n_instances):
            idx = np.argmax(min_dist)
            selected_indices.append(unlabeled_index[idx])
            top_k.append(min_dist[idx])

            x_new = X_pool[idx:idx+1, :]  
            used_data = np.append(used_data, x_new, axis=0)

            dist_new = cdist(X_pool, x_new, metric='euclidean').reshape(-1)
            min_dist = np.minimum(min_dist, dist_new)

            X_pool = np.delete(X_pool, idx, axis=0)
            unlabeled_index = np.delete(unlabeled_index, idx, axis=0)
            min_dist = np.delete(min_dist, idx, axis=0)

        return np.array(selected_indices), np.array(top_k)


    def query_QBC(self, unlabeled_samples, learner_list, al_batch_size):
        initial_pred=np.empty(shape=(len(learner_list), unlabeled_samples.shape[0]))
        for i in range(len(learner_list)):
            initial_pred[i,:] = learner_list[i].predict(unlabeled_samples).reshape(1, -1)
        utility = np.var(initial_pred, axis=0)
        selected_indices = self.multi_argmax(utility, n_instances=al_batch_size)
        top_k = utility[selected_indices]
        return selected_indices, top_k

#     source of the multi_argmax function:
#     @article{modAL2018,
#     title={mod{AL}: {A} modular active learning framework for {P}ython},
#     author={Tivadar Danka and Peter Horvath},
#     url={https://github.com/modAL-python/modAL},
#     note={available on arXiv at \url{https://arxiv.org/abs/1805.00979}}
# }
    
    def multi_argmax(self, values, n_instances):

        assert n_instances <= values.shape[0], 'n_instances must be less or equal than the size of utility.'

        max_idx = np.argpartition(-values, n_instances-1, axis=0)[:n_instances]
        return max_idx
    
    
    def QBC_retrain(self, idx, seed_QBC_bootstrap, member_list, used_data, used_label):
        key_early_stop = "RMSE"
        np.random.seed(seed_QBC_bootstrap[idx+1])
        new_list = []
        for m in member_list:
            jdx = np.random.choice(np.arange(used_data.shape[0]), used_data.shape[0], replace=True)
            m.fit(used_data[jdx], used_label[jdx], callbacks=[CustomEarlyStopCallback(key_early_stop)])
            new_list.append(m)
        return new_list


