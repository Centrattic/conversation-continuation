""" 
Here we extract all activations on all training data (40000 samples). 
Oh no dont do training data initially. It's more expensive and also maybe unncessary.
Let's just do input sequences from friend_hist.csv (12000 or so). We'll cache via hashing indexing, And we should also store who said what. 
Since this will help us attribute correctly to person (can filter out vectors from other person a bit)
We'll also extract last-layer activations particularly.

So we hash content and store in mmap with activations.
Then we have a json dict where each hash maps to AuthorID.

Then during sampling if tda flag is passed, we extract activations and check with get_cosine_similarity function.
And we first filter for all the hashes that belong to the correct author, and we load only those activations.

But maybe expensive to load. Perhaps we save lower dimensional projection (PCA). Maybe we can learn one too somehow?
Fine for now, Mistral model dimension isn't too big.

"""

