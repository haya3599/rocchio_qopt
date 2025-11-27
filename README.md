# rocchio_qopt
##  Overview
This project implements the **Rocchio algorithm** for computing the optimal query vector  
\( q_{opt} = 2\mu_R - \mu_{NR} \), based on TF-IDF document vectors.

The assignment required:
- Generating **50 documents** represented using Bag of Words + randomized TF-IDF
- Selecting **20% relevant** and **80% non-relevant** documents
- Computing μ_R, μ_NR, and q_opt
- Displaying the **top 5 most significant features** in q_opt
- Finding the **3 documents most similar** to q_opt using cosine similarity
