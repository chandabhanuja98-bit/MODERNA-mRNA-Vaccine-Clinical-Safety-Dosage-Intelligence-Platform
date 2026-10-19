import numpy as np
import matplotlib.pyplot as plt


vocab = ['A','C','G','U']
embed_dim = 8
np.random.seed(42)

embed_table = {
    base : np.random.randn(embed_dim)
    for base in vocab
}

def encode_sequence(sequence:str) -> np.ndarray :
    invalid_nodes = set(sequence)-set(vocab)
    if invalid_nodes :
        raise ValueError(
            f"Invalid nucleotide : {invalid_nodes}"
            f"Only A C G U are allowed"
        )
    return np.stack([embed_table[base] for base in sequence])

def softmax(score : np.ndarray) -> np.ndarray :
    score = score - np.max(score, axis=-1, keepdims=True)  # numerical stability
    exp_score = np.exp(score)
    return exp_score / np.sum(exp_score, axis=-1, keepdims=True)



def self_attention(sequence_embed:np.ndarray) -> np.ndarray:
    sequence_len,embed_dimension = sequence_embed.shape
    print(sequence_len,embed_dimension)

    w_q = np.random.randn(embed_dimension,embed_dimension)*0.1
    q = sequence_embed @ w_q 

    w_k = np.random.randn(embed_dimension,embed_dimension)*0.1
    k = sequence_embed @ w_k

    w_v = np.random.randn(embed_dimension,embed_dimension)*0.1
    v = sequence_embed @ w_v

    score = (q @ k.T) / np.sqrt(embed_dimension) 
    weights = softmax(score)
    output = weights @ v
    return weights,output

def plot_attention(weights,sequence,path) :
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(weights, cmap='viridis')
    ax.set_xticks(range(len(sequence)))
    ax.set_yticks(range(len(sequence)))
    ax.set_xticklabels(list(sequence))
    ax.set_yticklabels(list(sequence))
    ax.set_xlabel('Attending TO position')
    ax.set_ylabel('Attending FROM position')
    ax.set_title('mRNA Sequence Self-Attention Weights')
    fig.colorbar(im, label='Attention weight')
    fig.tight_layout()
    fig.savefig(path, dpi=150)



if __name__ == '__main__' :
    sequence = input("Enter mRNA sequence : ").strip().upper()
    print("Input sequence : ",sequence)

    sequence_embed = encode_sequence(sequence)

    weights,output = self_attention(sequence_embed)

    print("Attention weights : ", weights)
    print("Attention Output : ", output)

    assert np.allclose(weights.sum(axis=1), 1.0)

    plot_attention(weights,sequence,"outputs/figures")
    


   


  

    

    



