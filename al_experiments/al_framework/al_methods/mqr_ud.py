import numpy as np
import pandas as pd
from scipy.stats import entropy
from sklearn.cluster import kmeans_plusplus, KMeans
from sklearn.metrics import r2_score, mean_squared_error
import time
from scipy.spatial.distance import cdist

class ActiveLearner():
    
    def __init__(self, Q, K, n_queries, metric, initial_size, quantiles, batch_size, B, device):
        self.n_queries = n_queries
        self.Q = Q
        self.K = K
        self.metric = metric
        self.initial_size = initial_size
        self.quantiles = quantiles
        self.batch_size = batch_size
        self.device = device
        self.B = B
    
    # For each single AL experiment:
    def al_iterations(self, data_loader, X_train, X_test, y_train, y_test, seed_initial, seed_rf, evaluator, calculator):
        
        # Systems batches with corresponding label batches of training data
        sys_list = []
        label_list = []

        # unlabeled pool: samples grouped by system in the training data
        sys_list, label_list, _ = data_loader.append_systems(sys_list, label_list, X_train, y_train)
    
        # Total number of systems
        total_system = len(sys_list)
        print("Total number of data batches:", total_system)

        if type(data_loader).__name__ == "SemiConductor":
            X_test = X_test.drop("components", axis='columns')
            union_set_X = union_set_X.drop("components", axis='columns')
    
        # Feature dimensions
        X = X_test.shape[1]
        
        # Record indices of the unlabeled pool by systems: this is varying during the AL sampling process:
        X_index_list = []
        for s in np.arange(len(sys_list)):
            # Covert the dataframe of each system data into numpy array:
            sys = sys_list[s].to_numpy(dtype="float64")
            
            sys_list[s] = sys
            # The indices of samples in each system, which is used for recording the unlabelled pools, saved in X_index_list, ordered by the system list
            X_index_list.append(np.arange(sys.shape[0]))

        # Used data to record the labeled data during AL sampling
        used_data = np.empty(shape=(0, X))
        used_label = np.empty(shape=(0)).reshape(-1, 1)

        # Initialization Step:
        X_initial = np.empty(shape=(0, X))
        y_initial = np.empty(shape=(0)).reshape(-1, 1) 

        # Initialization ---> Randomly select one data batch to query and initialize the model:
        np.random.seed(seed_initial)
        idx = np.random.choice(range(total_system), size=1, replace=False)[0]
        
        selected_system = sys_list[idx]
        selected_system_label = label_list[idx]

        random_batch_s = selected_system.shape[0]

        # If the selected random batch has not enough samples for initilaization, or in superconductivity dataset, there are some different samples but have the same features: reinitialize the above process:
        while random_batch_s < self.initial_size or (np.unique(selected_system, axis=0).shape[0] == 1):
            # Redo-select another batch:
            idx = np.random.choice(range(total_system), size=1, replace=False)[0]
            # Recorded the selected batch:
            selected_system = sys_list[idx]
            selected_system_label = label_list[idx]
            random_batch_s = selected_system.shape[0]
            
        # Randomly select samples in the selected data batch
        np.random.seed(seed_initial)
        index = np.random.choice(range(random_batch_s), size=self.initial_size, replace=False)
        X_initial = selected_system[index]
        y_initial = selected_system_label[index].reshape(-1, 1)
        
        # When all randomly selected samples have same feature vectors, redo random sampling until the condition: Catboost cannot train with all repeated samples. 
        while (np.unique(X_initial, axis=0).shape[0] == 1):
            index = np.random.choice(range(random_batch_s), size=self.initial_size, replace=False)
            # Save the inital samples
            X_initial = selected_system[index]
            y_initial = selected_system_label[index].reshape(-1, 1)
                    
        used_data = np.append(used_data, X_initial, axis=0)
        used_label = np.append(used_label, y_initial, axis=0).reshape(-1, 1)

        # Update unlabeled pool:
        X_index_list[idx] = np.delete(X_index_list[idx], index, axis=0)
        
        model, prediction_ = calculator.quantile_regression(used_data, used_label, seed_rf, X_test, X)

        # Model performance after initialization:
        # For the whole test set
        prediction_point = evaluator.evaluation_metrics(prediction_, X_test, y_test, calculator.get_predictive_distribution, calculator.get_interval_ranges)
        
        testing_r2 = [r2_score(y_test, prediction_point)]
        testing_mse = [mean_squared_error(y_test, prediction_point)]

        print("Initialization R2:", testing_r2)
        print("Initialization MSE:", testing_mse)
        
        total_time = []
        acquisition_values = []
        acquisition_selected = []
        candidate_set_indices = []
        # AL Iterations:
        for idx in range(self.n_queries):
            
            start = time.process_time()
            print('Query no. %d' % (idx+1))
            # When the avaliable different batches < 5, stop AL selection.
            count_batch = 0
            for t in range(len(X_index_list)):
                if X_index_list[t].shape[0] >= self.batch_size:
                    count_batch += 1
            if count_batch < self.B:
                break
                    
            # Return: index of the batch, the indices in that batch, and acquisition scores of the selected samples
            top_indices, top_query_indices, utility_all, utility_selected, candidate_query_index = self.uncertainty_query(model, sys_list, X_index_list, self.batch_size, used_data, used_label, self.metric, self.K, self.Q, calculator, seed_initial)
            
            assert np.unique(np.asarray(top_indices)).shape[0] == self.B, "The selected Q data batches should be different."
            
            # Each query iteration, the acquisition values of each system:
            acquisition_values.append(utility_all)
            acquisition_selected.append(utility_selected)
            candidate_set_indices.append(candidate_query_index)
            
            end = time.process_time()
            total_time.append(end-start)
            # print("Time:", end-start)
            
            # Get the Multi_batches:
            for i in range(len(top_indices)):
                top_index = top_indices[i]
                top_query_index = top_query_indices[i]

                # The selected samples indices in the selected system
                sys_index = X_index_list[top_index][top_query_index]
                new_X = sys_list[top_index][sys_index]
                new_y = label_list[top_index][sys_index].reshape(-1, 1).ravel()
            
                # Update
                X_index_list[top_index] = np.delete(X_index_list[top_index], top_query_index, axis=0)

                # Adding the used data to the used_data pool
                used_data = np.append(used_data, new_X, axis=0)
                used_label = np.append(used_label, new_y.reshape(-1, 1), axis=0).reshape(-1, 1)

            print("Queried Samples:", np.unique(used_data, axis=0).shape[0])
            
            # Model performance after each query:
            model, prediction_ = calculator.quantile_regression(used_data, used_label, seed_rf, X_test, X)
            
            prediction_point = evaluator.evaluation_metrics(prediction_, X_test, y_test, calculator.get_predictive_distribution, calculator.get_interval_ranges)
            
            testing_r2.append(r2_score(y_test, prediction_point))
            testing_mse.append(mean_squared_error(y_test, prediction_point))
            
            print("R2:", r2_score(y_test, prediction_point))
            print("MSE:", mean_squared_error(y_test, prediction_point))
            
        testing_r2 = np.array(testing_r2)
        testing_mse = np.array(testing_mse)
        candidate_set_indices = np.array(candidate_set_indices, dtype=object)

        return testing_r2, testing_mse, used_data, used_label, acquisition_values, acquisition_selected

    
    def uncertainty_query(self, model, sys_list, X_index_list, batch_size, used_data, used_label, metric, K, Q, calculator, seed_initial):

        top_index, top_query_index, acquisition_values, acquisition_selected, candidate_query_index = self.custom_query_strategy(model, sys_list, X_index_list, batch_size, used_data, used_label, metric, K, Q, calculator, seed_initial)

        return top_index, top_query_index, acquisition_values, acquisition_selected, candidate_query_index
    
    
    def custom_query_strategy(self, model, sys_list, X_index_list, n_instances, used_data, used_label, metric, K, Q, calculator, seed_initial):
        assert metric in ["EP"], 'The uncertainty keywords must be "EP" metric.'
    
        # The analysis of the acquisition values in each system:
        top_all_utility_all = []
        top_all_utility_selected = []
        system_index = []

        candidate_query_index = []
        candidate_uncertainty_values = []

        # Each System:
        for sys_idx, sys in enumerate(sys_list):
            # Recording the AL scores for each system
            utility = []
            each_system = sys[X_index_list[sys_idx]]

            if each_system.shape[0] >= n_instances:
                # Intervals boundary
                sys_preds = model.predict(sys)
                
                range_min_value = np.min(sys_preds)
                range_max_value = np.max(sys_preds)
                ranges = calculator.get_interval_ranges(K, range_min_value, range_max_value)
                
                # Each sample in the system:
                for i in range(each_system.shape[0]):
                    prob_original = calculator.get_predictive_distribution(sys_preds[i, ], Q, K, ranges, range_max_value)
                    if metric == "EP":
                        utility.append(entropy(prob_original))

                # All the uncertainty scores of the system:
                utility = np.asarray(utility)
                top_all_utility_all.append(utility)
                # The indices of the candidate batches
                system_index.append(sys_idx)

                if np.unique(each_system, axis=0).shape[0] >= n_instances:
                    # K-means ++:
                    top_query_index, sample_acquisition = self.K_means_Plus_Plus(each_system, n_instances, utility, seed_initial)
                else:
                    top_query_index = self.multi_argmax(utility, n_instances=n_instances)
                    sample_acquisition = utility[top_query_index]
                
                assert np.unique(np.asarray(top_query_index)).shape[0] == n_instances, "The selected indices should be different."
                assert np.asarray(top_query_index).shape[0] == n_instances, "The selected samples should be equal to the n_instances."
                assert np.asarray(sample_acquisition).reshape(-1,1).shape[0] == n_instances, "The selected indices should be different."
                    
                candidate_query_index.append(top_query_index)
                candidate_uncertainty_values.append(np.mean(sample_acquisition))
                top_all_utility_selected.append(sample_acquisition)
                
            else:
                print("The System %s is not enough to be queried."% sys_idx)

        max_sys_idx = self.multi_argmax(np.asarray(candidate_uncertainty_values), self.B)
        system_index = np.asarray(system_index)
        candidate_query_index = np.asarray(candidate_query_index)
        
        selected_index = system_index[max_sys_idx]
        selected_top_query_index = candidate_query_index[max_sys_idx]

        return selected_index, selected_top_query_index, top_all_utility_all, top_all_utility_selected, candidate_query_index

    
    def K_means_Plus_Plus(self, seleceted_system, n_instances, top_all_utility, seed_initial):
        # K means++ 
        centers, center_id = kmeans_plusplus(seleceted_system, n_clusters=n_instances, n_local_trials=None, random_state=seed_initial)
        
        # calculate the distance of each sample to the centers
        distances = cdist(seleceted_system, seleceted_system[center_id])
        cluster_labels = np.argmin(distances, axis=1)
        
        # select the sample in the cluster that with the highest scores
        selected_indices = []
        selected_scores = []
        empty_clusters = []  # recording the empty cluster
        
        for i in range(n_instances):
            # Each cluster:
            cluster_points = np.where(cluster_labels == i)[0]
            if len(cluster_points) > 0:
                # get the sample with the highest score
                cluster_utilities = top_all_utility[cluster_points]
                best_point_idx = cluster_points[np.argmax(cluster_utilities)]
                selected_indices.append(best_point_idx)
                selected_scores.append(top_all_utility[best_point_idx])
            else:
                # record empty cluster
                empty_clusters.append(i)
        
        # if the empty cluster
        if empty_clusters:
            print("Empty clusters occur.")
            # remaining samples that have not been selected
            remaining_indices = list(set(range(len(seleceted_system))) - set(selected_indices))
            if remaining_indices:
                # select the samples with highest scores in the remaining samples
                remaining_utilities = top_all_utility[remaining_indices]
                n_additional = len(empty_clusters)  # number of samples need to be selected
                
                # select the top-k samples with the highest scores
                additional_best_indices = np.argsort(remaining_utilities)[-n_additional:]
                for idx in additional_best_indices:
                    selected_indices.append(remaining_indices[idx])
                    selected_scores.append(remaining_utilities[idx])
        
        selected_indices = np.array(selected_indices)
        sample_acquisition = np.array(selected_scores)
        
        # check the shape of returned values
        assert len(selected_indices) == n_instances, f"Selected {len(selected_indices)} points, but required {n_instances}"
        
        return selected_indices, sample_acquisition
    
    def multi_argmax(self, values, n_instances):
        #     source of the multi_argmax function:
        #     @article{modAL2018,
        #     title={mod{AL}: {A} modular active learning framework for {P}ython},
        #     author={Tivadar Danka and Peter Horvath},
        #     url={https://github.com/modAL-python/modAL},
        #     note={available on arXiv at \url{https://arxiv.org/abs/1805.00979}}
        # }
        assert n_instances <= values.shape[0], 'n_instances must be less or equal than the size of utility'

        max_idx = np.argpartition(-values, n_instances-1, axis=0)[:n_instances]
        return max_idx
