import numpy as np
import networkx as nx
import torch


class TokenToSequencePooler:
    def __init__(self, path_token_emb, path_attention_layers, alpha=0.85):
        # Initialize the pooler by loading token embeddings and attention layers from the given paths.
        # Handles the removal of CLS and END token embeddings from representations.

        self.path_token_emb = path_token_emb
        self.path_attention_layers = path_attention_layers
        self.representations_with_cls = self._load_torch_data(self.path_token_emb)
        self.alpha = alpha
        
        
        # remove the CLS and END token embedding
        if len(self.representations_with_cls.shape) == 2:
            self.representations =  self.representations_with_cls[1:-1]
        elif len(self.representations.shape) == 3:
            self.representations = self.representations_with_cls[:,1:-1, :]
        
        self.attn_all_layers = self._load_torch_data(self.path_attention_layers)

    def _load_torch_data(self, file_path, with_cls = False, verbose=False):
        # Load a torch tensor from the given file path.
        # If the file does not exist, print an error message and return None.

        try:
            tensors = torch.load(file_path).detach().numpy()
            return tensors
        except:
            print(f"There is no file named {file_path}")
            return None    


    def cls_pooling(self, save_path=None):
        # Extract the CLS token from the representations with CLS.
        # Optionally save the CLS token to the specified path.
        # If representations are not loaded, handle the error and return None.
        
        if self.representations_with_cls is not None:
            cls_token = self.representations_with_cls[0]
            if save_path:
                torch.save(save_path, cls_token)
            return cls_token.squeeze()
        print(f"representations_with_cls was None for sequence {self.uniprot_accession}", flush=True)
        return None  # Handle cases where CLS token is not available or representations are not loaded properly


    def create_pooled_matrices_across_layers(self, mtx_all_layers):
        # Perform max pooling across layers by selecting the maximum values across attention layers.
        # Returns the matrix after pooling the attention layers.
        mtx_max_of_max = torch.max(torch.from_numpy(mtx_all_layers[1]), dim=1)[0].numpy()
        return mtx_max_of_max
            
            

    def mean_pooling(self):
        # Perform mean pooling on the token representations by averaging across all tokens.

        if len(self.representations.shape) == 2:
            return np.mean(self.representations, axis=0)
        else:
            return np.mean(self.representations, axis=1)

    def max_pooling(self):
        # Perform max pooling on the token representations.
        
        if len(self.representations.shape) == 2:
            return np.max(self.representations, axis=0)
        else:
            return np.max(self.representations, axis=1)
    
    
    
    def pool_parti(self, 
                   verbose=False, 
                   return_importance=False,
                   alpha = None):
        # Perform pooling based on PageRank algorithm applied to attention matrices.
        # Optionally return importance weights or print details about the importance calculation.
        # Handles errors during the pooling process by printing detailed information.

        if len(self.attn_all_layers.shape) == 2 and self.attn_all_layers.shape[0] == self.attn_all_layers.shape[1]:
            print(f'You are using a single layer matrix of shape {self.attn_all_layers.shape}. This attention matrix may already be pooled across the heads.')
            matrix_to_pool = self.attn_all_layers
        else:
            matrix_to_pool = self.create_pooled_matrices_across_layers(mtx_all_layers=self.attn_all_layers).squeeze()

        

        dict_importance = self._page_rank(matrix_to_pool, 
                                         alpha = alpha)
        importance_weights = np.array(list(self._calculate_importance_weights(dict_importance).values()))
        
        if return_importance:
            return importance_weights
        if verbose:
            print(f'pagerank direct outcome is {dict_importance}\n')
            print(f'importance_weights dict of length {len(importance_weights)} looks like\n {importance_weights}')
            print(f'shape of the importance matrix is {len(importance_weights)} and for repr, its {self.representations.shape}')
            print(f"importance weights look like {sorted(importance_weights, reverse=True)[0:5]}")
            

        try:
            return torch.tensor(np.average(self.representations, weights=importance_weights, axis=0))
        except Exception as e:
            print(f"{e} in PageRank without cls", flush=True)
            print(f"This happened for protein {self.uniprot_accession}")
            print(f"self.representations shape {self.representations.shape}", flush=True)
            print(f"importance_weights {len(importance_weights)}", flush=True)
                
                

    def _page_rank(self, attention_matrix, personalization=None, nstart=None, 
                   prune_type="top_k_outdegree", 
                   alpha=None):
        # Run PageRank on the attention matrix converted to a graph.
        # Raises exceptions if the graph doesn't match the token sequence or has no edges.
        # Returns the PageRank scores for each token node.

        if alpha is None:
            alpha = self.alpha
        
        G = self._convert_to_graph(attention_matrix)
        
        if G.number_of_nodes() != attention_matrix.shape[0]:
            raise Exception(f"The number of nodes in the graph should be equal to the number of tokens in sequence! You have {G.number_of_nodes()} nodes for {attention_matrix.shape[0]} tokens.") 
        if G.number_of_edges() == 0:
            raise Exception(f"You don't seem to have any attention edges left in the graph.") 
        
        return nx.pagerank(G, alpha=alpha, tol=1e-06, weight='weight', personalization=personalization, nstart=nstart, max_iter=100)
        

    def _convert_to_graph(self, matrix):
        # Convert a matrix (e.g., attention scores) to a directed graph using networkx.
        # Each element in the matrix represents a directed edge with a weight.
        
        G = nx.from_numpy_array(matrix, create_using=nx.DiGraph)
    
        return G

    def _calculate_importance_weights(self, dict_importance):
        # Normalize the PageRank scores (importance values) so they sum to 1.
        # Exclude CLS and END token from the importance calculation.
        
        # Get the highest integer key
        highest_key = max(dict_importance.keys())
        
        # Remove the entry with the highest key (END) and the entry with key 0 (CLS token)
        del dict_importance[highest_key]
        del dict_importance[0]
    
        total = sum(dict_importance.values())
        return {k: v / total for k, v in dict_importance.items()}

