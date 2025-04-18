from keybert import KeyBERT
from sentence_transformers import SentenceTransformer
from sklearn.cluster import AgglomerativeClustering
import matplotlib.pyplot as plt
import networkx as nx

# load the text
def load_text(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()
    return text

# Step 1: Keyword Extraction
def extract_keywords(text, top_n=20):
    kw_model = KeyBERT()
    keywords = kw_model.extract_keywords(text, top_n=top_n, stop_words='english')
    return [kw[0] for kw in keywords]

# Step 2: Get Keyword Embeddings
def get_keyword_embeddings(keywords):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    return model.encode(keywords)

# Step 3: Cluster the Keywords and Get Cluster Names
def cluster_keywords_with_labels(keywords, n_clusters=3):
    embeddings = get_keyword_embeddings(keywords)
    clustering_model = AgglomerativeClustering(n_clusters=n_clusters)
    cluster_labels = clustering_model.fit_predict(embeddings)

    clustered = {}
    cluster_names = {}

    for idx, label in enumerate(cluster_labels):
        keyword = keywords[idx]
        clustered.setdefault(label, []).append(keyword)

    # Choose first keyword in each cluster as the label (or modify to get centroid-representative)
    for label, kw_list in clustered.items():
        cluster_names[label] = kw_list[0]  # Simple method — pick first keyword

    return clustered, cluster_names

# hierarchical mind map generator
def generate_hierarchical_mind_map(main_topic, clustered_keywords, cluster_names):
    G = nx.Graph()
    G.add_node(main_topic)

    for cluster_id, keywords in clustered_keywords.items():
        category = cluster_names[cluster_id]  # Use keyword-based cluster name
        G.add_node(category)
        G.add_edge(main_topic, category)

        for keyword in keywords:
            if keyword != category:  # Avoid self-loop
                G.add_node(keyword)
                G.add_edge(category, keyword)

    plt.figure(figsize=(14, 10))
    pos = nx.spring_layout(G, k=0.5)
    nx.draw(G, pos, with_labels=True, node_size=2000, node_color='lightblue', edge_color='gray', font_size=10, font_weight='bold')
    plt.title(f"Mind Map for: {main_topic}", fontsize=16)
    plt.axis('off')
    plt.show()

# main pipeline
def run_mind_map_pipeline(text, main_topic, top_n=20, n_clusters=3):
    keywords = extract_keywords(text, top_n=top_n)
    clustered_keywords, cluster_names = cluster_keywords_with_labels(keywords, n_clusters=n_clusters)
    generate_hierarchical_mind_map(main_topic, clustered_keywords, cluster_names)

# Example usage
if __name__ == "__main__":
    input_text = load_text("text.txt")
    main_topic = input("Enter main topic: ")  # e.g., "Free Fall"
    run_mind_map_pipeline(input_text, main_topic)
