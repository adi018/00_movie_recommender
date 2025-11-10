from recommender import BookRecommender

if __name__ == '__main__':
    r = BookRecommender()
    recs = r.recommend('I want a gripping mystery novel with clever detective work', k=3)
    for i, x in enumerate(recs, 1):
        print(f"{i}. {x['title']} - {x['authors']} ({x['published_year']}) score={x['score']:.4f}")
