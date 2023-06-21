# Graph based Issue Detection

Graph based Issue Detection is a project that aims to analyze different problematics present in news articles using natural language processing (NLP) techniques and graphs. The project focuses on various clusters dedicated to specific sectors such as health, economy, tourism, crime, and more.

## Research Question

Is there a lexical relationship between two or more clusters that discuss different sectors?

## Hypothesis

It is expected that there is a lexical connection between clusters that frequently appear in both sectors.

## Methodology

The following steps are involved in verifying the hypothesis:

1. **Data Collection**: Gather a comprehensive set of news articles related to various sectors of interest.

2. **Natural Language Processing (NLP)**: Apply NLP techniques to analyze the texts and extract relevant keywords. This includes tokenization, removal of stop words, lemmatization, and other preprocessing steps.

3. **Word Frequency Analysis**: Calculate the frequency of occurrence of words in the policy and crime clusters separately. Identify the most frequent words in each cluster.

4. **Keyword Comparison**: Identify the keywords mentioned in the research question and check their presence and frequency in both clusters.

5. **Co-occurrence Analysis**: Analyze the co-occurrence of keywords in both clusters to determine if there is a significant lexical relationship. Construct a weighted undirected graph based on the identified keywords and their frequencies.

6. **Graph Analysis**: Utilize graph algorithms such as community detection and centrality calculations to identify subsets of nodes with stronger connections. This helps identify communities or clusters that contain the mentioned keywords.

## Repository Name: NewsGraphAnalyzer

This repository contains the code and resources for the News Cluster Analyzer project. It implements NLP techniques and graph analysis to explore lexical relationships between different clusters of news articles.

## Getting Started

To get started with the News Cluster Analyzer, follow these steps:

1. Clone the repository: `https://github.com/recmera/graph-based-issue-detection`
2. Install the required dependencies: `pip install -r requirements.txt`
3. Customize the notebook file to adapt it to your specific needs.

## Contributing

Contributions to the News Cluster Analyzer project are welcome! If you find any issues or have suggestions for improvements, feel free to open a pull request or submit an issue in the GitHub repository.

## License

This project is licensed under the [MIT License](LICENSE).

## Contact

For any inquiries or further information, please contact [rec.mera@gmail.com](mailto:rec.mera@gmail.com).

