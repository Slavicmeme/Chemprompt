import os
import random
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import pearsonr, spearmanr
from tqdm import tqdm


class genetic:
    """
    Genetic Algorithm for descriptor subset optimization.
    Works with a filtered descriptor space (e.g., 120 descriptors after global filtering),
    and performs search only on the subset selected via feature selection (e.g., 50 features).
    """

    def __init__(
        self,
        smiles_list,
        y,
        llm_model,
        dataset_name,
        selected_indices,
        num_generations=10,
        pop_size=10,
        top_k=3,
        mutation_rate=0.01,
        save_predictions=False
    ):
        self.smiles_list = smiles_list
        self.y = y
        self.llm = llm_model
        self.dataset_name = dataset_name
        self.selected_indices = selected_indices  # feature selection indices (filtered descriptor space)
        self.num_features = len(selected_indices)

        self.num_generations = num_generations
        self.pop_size = pop_size
        self.top_k = top_k
        self.mutation_rate = mutation_rate
        self.save_predictions = save_predictions

        self.device = llm_model.device
        self.population = self._init_population()
        self.seen_individuals = set(tuple(ind) for ind in self.population)

        self.individual_metrics = {}
        self.metrics_dir = os.path.join("./evolutionary_process", self.dataset_name)
        os.makedirs(self.metrics_dir, exist_ok=True)
        self.population_history = []
        self.best_history = []

    # -------------------------------
    # Population Initialization
    # -------------------------------
    def _init_population(self):
        pop = []
        while len(pop) < self.pop_size:
            ind = [random.choice([True, False]) for _ in range(self.num_features)]
            pop.append(ind)
        return pop

    # -------------------------------
    # Mutation and Crossover
    # -------------------------------
    def _mutate(self, individual):
        """
        Perform swap mutation with a given probability.
        Keeps the number of active (True) features constant.
        """
        # Skip mutation if the random draw exceeds mutation rate
        if random.random() > self.mutation_rate:
            return individual
    
        # Identify active and inactive feature indices
        indices_on = [i for i, v in enumerate(individual) if v]
        indices_off = [i for i, v in enumerate(individual) if not v]
    
        # Swap one active feature with one inactive feature
        if indices_on and indices_off:
            i_off = random.choice(indices_off)
            i_on = random.choice(indices_on)
            individual[i_off] = True
            individual[i_on] = False
    
        return individual

    def _crossover(self, parents):
        """
        Perform single-point crossover between parent individuals.
        Each child inherits a portion of genes from p1 and the rest from p2.
        Mutation is applied probabilistically after crossover.
        """
        children = []
        while len(parents) + len(children) < self.pop_size:
            # Randomly select two distinct parents
            p1, p2 = random.sample(parents, 2)
    
            # Choose a random crossover point (1 to num_features - 1)
            split_point = random.randint(1, self.num_features - 1)
    
            # Inherit genes from p1 (front) and p2 (back)
            child = p1[:split_point] + p2[split_point:]
    
            # Apply mutation with a given probability
            if random.random() < self.mutation_rate:
                child = self._mutate(child)
    
            # Prevent duplicate individuals in the population
            key = tuple(child)
            if key not in self.seen_individuals:
                self.seen_individuals.add(key)
                children.append(child)
    
        return parents + children

    # -------------------------------
    # Evaluation
    # -------------------------------
    def _evaluate_population(self, embeddings_list):
        kf = KFold(n_splits=3, shuffle=True, random_state=42)
        scores = []

        for idx, individual in enumerate(self.population):
            key = tuple(individual)
            if key not in self.individual_metrics:
                X = embeddings_list[idx]
                r2s, rmses, pccs, sps = [], [], [], []

                for fold_idx, (tr, te) in enumerate(kf.split(X), 1):
                    X_tr, X_te = X[tr], X[te]
                    y_tr, y_te = self.y[tr], self.y[te]
                    model = Ridge().fit(X_tr, y_tr)
                    y_pred = model.predict(X_te)

                    r2s.append(r2_score(y_te, y_pred))
                    rmses.append(np.sqrt(mean_squared_error(y_te, y_pred)))
                    pccs.append(pearsonr(y_te.ravel(), y_pred.ravel())[0])
                    sps.append(spearmanr(y_te.ravel(), y_pred.ravel())[0])

                    if self.save_predictions:
                        pred_dir = os.path.join(self.metrics_dir, f"generation_{self.current_gen}", f"ind_{idx+1}")
                        os.makedirs(pred_dir, exist_ok=True)
                        pd.DataFrame({
                            "y_true": y_te.ravel(),
                            "y_pred": y_pred.ravel()
                        }).to_csv(os.path.join(pred_dir, f"fold_{fold_idx}.csv"), index=False)

                self.individual_metrics[key] = {
                    "r2": np.mean(r2s),
                    "rmse": np.mean(rmses),
                    "pcc": np.mean(pccs),
                    "spearman": np.mean(sps)
                }

            m = self.individual_metrics[key]
            flag_str = "".join("1" if b else "0" for b in individual)
            print(f"Ind {idx+1}: R²={m['r2']:.3f} | flag={flag_str}")
            scores.append(m["r2"])

        return scores

    # -------------------------------
    # Save utilities
    # -------------------------------
    def _save_generation_metrics(self):
        gen_records = []
        for idx, ind in enumerate(self.population):
            fstr = "".join("1" if b else "0" for b in ind)
            m = self.individual_metrics[tuple(ind)]
            gen_records.append({
                "generation": self.current_gen,
                "ind_idx": idx + 1,
                "flag": fstr,
                **m
            })
        df_gen = pd.DataFrame(gen_records)
        df_gen.to_csv(os.path.join(self.metrics_dir, f"population_gen_{self.current_gen}.csv"), index=False)
        self.population_history.append(df_gen)

        best_row = df_gen.loc[df_gen["r2"].idxmax()].to_dict()
        pd.DataFrame([best_row]).to_csv(os.path.join(self.metrics_dir, f"best_gen_{self.current_gen}.csv"), index=False)
        self.best_history.append(best_row)

    def _save_summary(self):
        if self.population_history:
            pd.concat(self.population_history, ignore_index=True).to_csv(
                os.path.join(self.metrics_dir, "population_all_generations.csv"), index=False
            )
        if self.best_history:
            pd.DataFrame(self.best_history).to_csv(
                os.path.join(self.metrics_dir, "best_each_generation.csv"), index=False
            )

    # -------------------------------
    # Run Genetic Algorithm
    # -------------------------------
    def run(self):
        best_individual, best_score = None, -np.inf
        no_improve = 0
        desc_count = len(self.llm.property_names)  # filtered descriptor count (e.g., 120)

        for gen in range(1, self.num_generations + 1):
            self.current_gen = gen
            print(f"\n=== Generation {gen}/{self.num_generations} ===")

            embeddings_list = []
            for i, ind in enumerate(self.population):
                # Convert GA flag (K-dim) → filtered full flag (desc_count)
                full_flag = [False] * desc_count
                for j, idx in enumerate(self.selected_indices):
                    if ind[j]:
                        full_flag[idx] = True

                embed = self.llm.get_embeddings_for_ga(
                    self.smiles_list,
                    flag=full_flag,
                    save_dir=os.path.join(self.metrics_dir, f"generation_{gen}", "embeddings")
                )
                embeddings_list.append(embed)

            scores = self._evaluate_population(embeddings_list)
            self._save_generation_metrics()

            gen_best_idx = int(np.argmax(scores))
            gen_best_score = scores[gen_best_idx]
            if gen_best_score > best_score:
                best_score = gen_best_score
                best_individual = self.population[gen_best_idx]
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= 10:
                    print("[EarlyStop] 10 generations without improvement")
                    break

            top_k_idx = np.argsort(scores)[-self.top_k:]
            parents = [self.population[i] for i in top_k_idx]
            self.population = self._crossover(parents)

        self._save_summary()

        # Final: GA best flag (K-length) → full flag (filtered descriptor space)
        full_flag = [False] * desc_count
        for j, idx in enumerate(self.selected_indices):
            if best_individual[j]:
                full_flag[idx] = True

        best_flag_str = "".join("1" if b else "0" for b in best_individual)
        full_flag_str = "".join("1" if b else "0" for b in full_flag)
        best_metrics = self.individual_metrics[tuple(best_individual)]

        pd.DataFrame([{
            "flag": best_flag_str,
            "full_flag": full_flag_str,
            **best_metrics
        }]).to_csv(os.path.join(self.metrics_dir, "best_overall_metrics.csv"), index=False)

        print(f"\nBest overall R²={best_score:.3f} | flag={best_flag_str}")
        self.best_flag_str = best_flag_str

        return best_individual, full_flag