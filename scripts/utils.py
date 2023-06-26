import re
from collections import Counter

import networkx as nx
import matplotlib.pyplot as plt


def clean_title(df):
    for index, row in df.iterrows():
        title = row['title']
        split_title = title.split('-')
        cleaned_title = split_title[0].strip()  # Elimina espacios en blanco alrededor del texto
        df.at[index, 'title'] = cleaned_title
        return df

def get_representative_description(doc):
    pattern = r"'([^']*)'"
    matches = re.search(pattern, doc)
    if matches:
        return matches.group(1)
    return None


def create_graph(topic_df):
    graph = nx.Graph()
    titles_keywords = {}

    for index, row in topic_df.iterrows():
        title = row['id_news']
        keywords = set(extract_keywords(row['keywords']))
        titles_keywords[title] = keywords
        graph.add_node(title)

    edges = [(title1, title2, {'keywords': keywords1.intersection(keywords2)})
             for title1, keywords1 in titles_keywords.items()
             for title2, keywords2 in titles_keywords.items()
             if title1 != title2 and keywords1.intersection(keywords2)]

    graph.add_edges_from(edges)
    return graph

def determine_edge_weights(graph):
    edge_weights = nx.get_edge_attributes(graph, 'keywords')
    nx.set_edge_attributes(graph, {edge: len(keywords) for edge, keywords in edge_weights.items()}, 'weight')

def visualize_graph(graph, topic_df, clusters, cluster_num):
    plt.figure(figsize=(12, 9))
    pos = nx.kamada_kawai_layout(graph)

    # Calculate degree centrality for node sizes
    node_sizes = [500 * nx.degree_centrality(graph)[node] for node in graph.nodes()]

    # Customize node and label styles
    node_colors = ['skyblue' if node in topic_df['id_news'].values else 'lightgray' for node in graph.nodes()]
    node_border_color = ['black' if node in topic_df['id_news'].values else 'gray' for node in graph.nodes()]

    edge_weights = nx.get_edge_attributes(graph, 'weight')
    edge_labels = nx.get_edge_attributes(graph, 'keywords')
    if edge_weights:  # Check if there are edge weights
        edge_colors = list(edge_weights.values())  # Get the edge weights

        nx.draw_networkx_edges(graph, pos, alpha=0.5, edge_cmap=plt.cm.Blues, edge_vmin=min(edge_colors), edge_vmax=max(edge_colors))
        nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels, font_size=6, font_color='black')
    
    nx.draw_networkx_nodes(graph, pos, node_size=node_sizes, node_color=node_colors, edgecolors=node_border_color, alpha=0.8)
    nx.draw_networkx_labels(graph, pos, font_size=8, font_weight='normal')

    # Add colorbar for edge weights
    if edge_weights:  # Check if there are edge weights
        sm = plt.cm.ScalarMappable(cmap=plt.cm.Blues, norm=plt.Normalize(vmin=min(edge_colors), vmax=max(edge_colors)))
        plt.colorbar(sm)

    plt.title("Grafo de keywords compartidos entre noticias del mismo clúster - Clúster: {}".format(clusters.loc[clusters.Topic == cluster_num].Representation))
    plt.axis('off')
    plt.show()

def shared_keywords_of_news_in_cluster(df, clusters, cluster_num):
    topic_df = df[df['Topic'] == cluster_num]
    graph = create_graph(topic_df)
    determine_edge_weights(graph)
    visualize_graph(graph, topic_df, clusters, cluster_num)
    
def sort_nodes_by_degree_centrality(graph):
    # Calcula el grado de centralidad de cada nodo
    degree_centralities = nx.degree_centrality(graph)
    
    # Ordena los nodos según su grado de centralidad de forma descendente
    sorted_nodes = sorted(degree_centralities, key=degree_centralities.get, reverse=True)
    return sorted_nodes
    
    
def extract_keywords(string):
    pattern = r"'([^']*)'"
    matches = re.findall(pattern, string)
    return matches



from collections import Counter

def generate_graph_from_dataframe(dataframe):
    # Crear el grafo
    graph = nx.Graph()

    # Iterar sobre los tópicos del DataFrame
    for topic, keywords_list in zip(dataframe['Topic'], dataframe['keywords']):
        # Crear una lista única de todas las keywords de la lista de keywords
        all_keywords = []
        for keywords in keywords_list:
            all_keywords.append(keywords)
        

        # Eliminar duplicados
        unique_keywords = list(set(all_keywords))        
        unique_keywords = [keyword for keyword in unique_keywords if len(keyword) > 7]

        # Agregar el nodo del tópico al grafo
        graph.add_node(topic, keywords=unique_keywords)

        # Conectar el tópico con los tópicos existentes que tienen keywords en común
        for existing_topic, existing_keywords in graph.nodes(data='keywords'):
            if existing_topic != topic:  # Evitar conexión consigo mismo
                common_keywords = set(unique_keywords).intersection(existing_keywords)
                if common_keywords:
                    graph.add_edge(topic, existing_topic, keywords=common_keywords)

    # Visualizar el grafo
    pos = nx.kamada_kawai_layout(graph)
    nx.draw(graph, pos, with_labels=True, node_color='skyblue', node_size=500, font_size=8)
    edge_labels = nx.get_edge_attributes(graph, 'keywords')
    nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels, font_size=6)

    # Mostrar el grafo
    plt.title("Relación de tópicos mediante keywords")

def keyword_distribution(dataframe):    
    merged_list = []

    for sublist in dataframe.keywords.tolist():
        merged_list.extend(sublist)


    # Calcular la frecuencia de las palabras
    word_frequency = Counter(merged_list)

    # Filtrar las keywords cuyo contador es igual a 1
    filtered_keywords = {word: count for word, count in word_frequency.items() if count > 1}

    # Ordenar las palabras y sus frecuencias en orden descendente
    sorted_word_frequency = sorted(filtered_keywords.items(), key=lambda x: x[1], reverse=True)

    # Extraer las palabras y sus frecuencias ordenadas
    words = [word for word, freq in sorted_word_frequency]
    frequencies = [freq for word, freq in sorted_word_frequency]

    # Graficar la distribución de palabras
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(words)), frequencies)
    plt.xlabel('Palabras')
    plt.ylabel('Frecuencia')
    plt.xticks(range(len(words)), words, rotation=90)
    plt.title('Distribución de palabras por frecuencia')
    
    
def community_louvain(graph):
    
    import community
    from community import community_louvain

    # Calcular la partición utilizando el algoritmo de Louvain
    partition = community_louvain.best_partition(graph)

    # Convertir la partición en un diccionario
    partition_dict = {}
    for node, comm in partition.items():
        partition_dict[node] = comm

    # Calcular la modularidad
    modularity = community.modularity(partition, graph)

    # Visualizar el grafo con la partición coloreada
    pos = nx.spring_layout(graph)  # Posiciones de los nodos para el layout
    cmap = plt.get_cmap("tab10")  # Mapa de colores para las comunidades
    plt.figure(figsize=(10, 6))  # Tamaño de la figura

    # Dibujar los nodos de cada comunidad con colores diferentes
    for community_id in set(partition_dict.values()):
        nodes = [node for node, comm_id in partition_dict.items() if comm_id == community_id]
        nx.draw_networkx_nodes(
            graph,
            pos,
            nodelist=nodes,
            node_color=cmap(community_id),
            node_size=200,
            alpha=0.8,
        )

    # Dibujar las aristas del grafo
    nx.draw_networkx_edges(graph, pos, alpha=0.5)

    # Mostrar etiquetas de los nodos
    nx.draw_networkx_labels(graph, pos, font_size=8, font_color="black")

    # Mostrar información de la modularidad en el título
    plt.title("Red con Modularidad: {:.3f}".format(modularity))

    # Ocultar ejes
    plt.axis("off")

    # Mostrar el gráfico
    plt.show()
    
    return partition_dict
