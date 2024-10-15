import pandas as pd
import numpy as np  # Numpy for most math, including linear algebra
import matplotlib.pyplot as plt  # Matplotlib for plotting

# Input: 2 movies m1 and m2
# Returns a list of ratings where m1 and m2 have the same users
def createSimilarList(m1, m2, dict_movies):
    inter = list(set(dict_movies.get(m1)) & set(dict_movies.get(m2)))
    i=0
    range=len(inter)
    out=[]
    while i<range:
        out.append(dict_movies[m1][inter[i]].get('movie_rating'))
        i=i+1
    return out
        
#Calculates the adjusted cosine similarity
def adjusted_cosine_similarity(rating1, rating2):
    arr1 = np.array(rating1)
    arr2 = np.array(rating2)
    
    product = np.dot(arr1, arr2)
    magnitude1 = np.linalg.norm(arr1)
    magnitude2 = np.linalg.norm(arr2)

    return product / (magnitude1 * magnitude2)

# Input movie, output the movie that is most similar using adjusted cosine similarity
def findMostSimilar(mov1, dict_movies):
    mov2=1
    coefficients=[]
    for x in dict_movies:
        list1=createSimilarList(mov1,mov2, dict_movies)
        list2=createSimilarList(mov2,mov1, dict_movies)
        mov2=mov2+1
        coefficients.append(adjusted_cosine_similarity(list1, list2))
    m=max(coefficients)
    best=[coefficients.index(m), m]
    return best

def compute_similarity(input_file, output_file, threshold=5):
    """
    Function to compute similarity scores
    
    Arguments
    ---------
    input_file: str, path to input MovieLens file 
    output_file: str, path to output .csv 
    user_threshold: int, optional argument to specify
    the minimum number of common users between movies 
    to compute a similarity score. The default value 
    should be 5. 
    """
    df = pd.read_csv(input_file, names=["user_id", "movie_id", "movie_rating", "timestamp"], sep='\t')
    df.drop('timestamp',axis=1, inplace=True)
    dict_movies = df.groupby('movie_id')[['user_id', 'movie_rating']].apply(lambda x: x.set_index('user_id').to_dict(orient='index')).to_dict()

    k=1
    mostSimilarMovies=[]
    dict_length=len(dict_movies)
    while k < dict_length+1:
        mostSimilarMovies.append(findMostSimilar(k, dict_movies))
        k=k+1

    d_ratings = df.groupby('movie_id')['movie_rating'].apply(lambda x: list(x)).to_dict()
    df_ratings = pd.DataFrame(list(d_ratings.items()), columns=['movie_id', 'movie_ratings'])

    movierat = pd.DataFrame(mostSimilarMovies, columns =['Most_Similar_Movie', 'Coefficient']) 
    final=pd.concat([df_ratings, movierat], axis=1)
    final['movie_ratings'] = final['movie_ratings'].apply(lambda x: 0 if len(x) < threshold else x)
    final['Most_Similar_Movie'] = final.apply(lambda row: "NaN" if row['movie_ratings'] ==0 else row['Most_Similar_Movie'], axis=1)
    final.drop('movie_ratings',axis=1, inplace=True)

    final[:20].to_csv(output)


if __name__ == "__main__":
    input='https://raw.githubusercontent.com/jpelayo35/assignment2/refs/heads/main/u.data'
    output='DATA5100/assignment2/df_abridged.csv'
    compute_similarity(input,output)