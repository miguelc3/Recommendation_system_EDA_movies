import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import warnings

warnings.filterwarnings('ignore')


# Function to retrieve recommendations
def recommend_titles(title, similarity_matrix, dataframe):
    try:
        # Find the index of the title
        title_idx = dataframe.index[dataframe['title'] == title].tolist()[0]

        # Get the similarity scores for the title in the sim matrix
        similarity_scores = similarity_matrix[title_idx]

        # Sort scores -> not retrive the title asked itself
        sorted_indices = similarity_scores.argsort()[::-1][1:6]

        
        recommendations = dataframe['title'].iloc[sorted_indices].tolist()

        return recommendations
    except IndexError:
        return "Title not found."


def create_similarity_matrix(data, text_feature):
    # Initialize the TF-IDF Vectorizer for english words
    vectorizer = TfidfVectorizer(stop_words='english')

    # Fit and transform the new text feature to create the TF-IDF matrix
    tfidf_matrix = vectorizer.fit_transform(data[text_feature])

    # Calculate the cosine similarity matrix between all 'texts '  
    cosine_sim_matrix = cosine_similarity(tfidf_matrix)

    return cosine_sim_matrix
    

def main():

    # Load the dataset
    data_path = r'./dataset.csv'
    data = pd.read_csv(data_path)

    # Fill NaN values in specified columns with an empty string
    for col in ['director', 'cast', 'rating']:
        data[col].fillna('', inplace=True)

    # Create the text feature
    data['text'] = data[['title', 'director', 'cast', 'country', 'release_year', 
                     'rating', 'listed_in', 'description']].astype(str).agg(' '.join, axis=1)

    # Create the similarity matrix
    cosine_sim_matrix = create_similarity_matrix(data, 'text')

    # while loop to ask the user titles to get simmilar recommendations
    while 1:
        print('\n')
        title = input('Reccomendations for movies simmilar with: ')
        print(recommend_titles(title, cosine_sim_matrix, data))
        print('\n')

        c = input('Do you want to search another title? ("y" or "n") ')
    
        if c == 'n':  # break the loop if the user don't want to continue
            break


if __name__ == '__main__':
    main()

    