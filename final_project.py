import json
import string
import statistics
import math
import gender_guesser.detector as gender
import matplotlib.pyplot as plt
from collections import Counter
from scipy import stats
from stop_words import get_stop_words
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import pandas as pd
import seaborn as sns

stop_words = get_stop_words('en')
d = gender.Detector(case_sensitive=False)
nltk.download('vader_lexicon') # Download the VADER lexicon resource
nltk.download('punkt')
translator = str.maketrans(string.punctuation, ' ' * len(string.punctuation))

# print(d.get_gender("None"))
# print(d.get_gender(u"William"))
# file = 'AMAZON_FASHION_5.json'

# DATASETS FOUND HERE: https://cseweb.ucsd.edu/~jmcauley/datasets/amazon_v2/
# Download "Small" subsets for experimentation -> 5-core 

file = 'Movies_and_TV_5.json'

# data format: (rating, gender, reviewText)
data = []
# alt_data formate: asin : [(rating, gender), ...]
alt_data = {}

with open(file, 'r') as fp:
    for line in fp:
        # Pre process names, remove punc and get first name
        review_dict = json.loads(line.strip())
        name = review_dict.get('reviewerName', "None")
        first_name = name.translate(translator).partition(" ")[0]
        # Remove all data with unknown gender
        reveiwer_gender = d.get_gender(first_name)
        if reveiwer_gender == 'mostly_male' or reveiwer_gender == 'male':
            reveiwer_gender = 'M'
        elif reveiwer_gender == 'mostly_female' or reveiwer_gender == 'female':
            reveiwer_gender = 'F'
        elif reveiwer_gender == 'andy':
            reveiwer_gender = 'A'
        if reveiwer_gender != 'unknown':
            data.append([float(review_dict['overall']), reveiwer_gender, review_dict['reviewText'] if 'reviewText' in review_dict else ""])
            if review_dict['asin'] not in alt_data:
                alt_data[review_dict['asin']] = []
            alt_data[review_dict['asin']].append([float(review_dict['overall']), reveiwer_gender])

print(f"Size of data collected: {len(data)}")
print(f"Number of males: {sum(1 for row in data if row[1] == 'M')}")
print(f"Number of females: {sum(1 for row in data if row[1] == 'F')}")
print(f"Number of androgynous: {sum(1 for row in data if row[1] == 'A')}")

#ISHA
male_ratings = [row[0] for row in data if row[1] == 'M']
female_ratings = [row[0] for row in data if row[1] == 'F']
andro_ratings = [row[0] for row in data if row[1] == 'A']

print(f"Average rating of males: {statistics.fmean(male_ratings)}")
print(f"Average rating of females: {statistics.fmean(female_ratings)}")
print(f"Average rating of androgynous: {statistics.fmean(andro_ratings)}")

def visualize_distribution():
    # ISHA analyzing distribution of review ratings for each gender group 
    male_rating_counts = [sum(1 for rating in male_ratings if rating == i) for i in range(1, 6)]
    female_rating_counts = [sum(1 for rating in female_ratings if rating == i) for i in range(1, 6)]
    andro_rating_counts = [sum(1 for rating in andro_ratings if rating == i) for i in range(1, 6)]

    # isha visualizing the distribution of review ratings 
    plt.figure(figsize=(10,6))
    x = range(1,6)
    width = 0.25
    plt.bar([i - width for i in x], male_rating_counts, width, label = 'Male')
    plt.bar(x, female_rating_counts, width, label = 'Female')
    plt.bar([i + width for i in x], andro_rating_counts, width, label = 'Androgynous')
    plt.xlabel('Rating')
    plt.ylabel('Count')
    plt.title('Distribution of Review Ratings by Gender')
    plt.legend()
    plt.xticks(x)
    plt.show()
    
def t_test():
    # ISHA t - test to quanity the significance of gender differences 
    t_stat, p_value = stats.ttest_ind(male_ratings, female_ratings)
    print(f"T-test results:")
    print(f"T-statistic: {t_stat}")
    print(f"P-value: {p_value}")

    # ISHA now we need to analyze gender bias across different product categories visualzie
    plt.figure(figsize=(8, 6))
    plt.hist(male_ratings, bins=20, alpha=0.5, label='Male', color='blue')
    plt.hist(female_ratings, bins=20, alpha=0.5, label='Female', color='red')
    plt.axvline(statistics.fmean(male_ratings), color='blue', linestyle='dashed', linewidth=2)
    plt.axvline(statistics.fmean(female_ratings), color='red', linestyle='dashed', linewidth=2)
    plt.xlabel('Rating')
    plt.ylabel('Frequency')
    plt.title(f"T-test Results\nT-statistic: {t_stat:.2f}, P-value: {p_value:.4f}")
    plt.legend()
    plt.show()

def diff_ratings_per_prodct():
    # Diff between male and female ratings per product
    alt_data_diff = []

    for key in alt_data:
        total_male_rating = sum(pair[0] for pair in alt_data[key] if pair[1] == 'M')
        total_male_count = sum(1 for pair in alt_data[key] if pair[1] == 'M')
        total_female_rating = sum(pair[0] for pair in alt_data[key] if pair[1] == 'F')
        total_female_count = sum(1 for pair in alt_data[key] if pair[1] == 'F')
        avg_male_rating = total_male_rating / (total_male_count or 1)
        avg_female_rating = total_female_rating / (total_female_count or 1)

        if total_male_count == 0 or total_female_count == 0:
            # alt_data_diff.append(0.0)
            pass
        else:
            alt_data_diff.append(abs(avg_female_rating - avg_male_rating))

    print(f"Average diff between M and F ratings per product is: {statistics.fmean(alt_data_diff)}")
    # print(alt_data_diff)

def word_frequency():

    # male_review_texts = [row[2] for row in data if row[1] == 'M']
    # female_review_texts = [row[2] for row in data if row[1] == 'F']

    male_review_words = []
    female_review_words = []

    for row in data:
        if row[1] == 'M':
            male_review_words += filter(lambda i: i not in stop_words, row[2].translate(str.maketrans('', '', string.punctuation)).lower().split())
        elif row[1] == 'F':
            female_review_words += filter(lambda i: i not in stop_words, row[2].translate(str.maketrans('', '', string.punctuation)).lower().split())

    male_word_freq = Counter(male_review_words)
    female_word_freq = Counter(female_review_words)


    sorted_male_word_freq = sorted(male_word_freq.items(), key= lambda x: x[1], reverse=True)
    sorted_female_word_freq = sorted(female_word_freq.items(), key= lambda x: x[1], reverse=True)

    NUM_TOP_RESULTS = 25
    top_male_words = sorted_male_word_freq[:NUM_TOP_RESULTS]
    top_female_words = sorted_female_word_freq[:NUM_TOP_RESULTS]

    print("TOP MALE WORDS BY FREQUENCY: ")
    for word in top_male_words:
        print(f"{word[0]}: {word[1]}")

    print("TOP FEMALE WORDS BY FREQUENCY: ")
    for word in top_female_words:
        print(f"{word[0]}: {word[1]}")

def sentiment_analysis():

    # Define lists to store positive and negative reviews
    positive_reviews_male = []
    positive_reviews_female = []
    negative_reviews_male = []
    negative_reviews_female = []
    sid = SentimentIntensityAnalyzer()

    for number,row in enumerate(data):
        overall_rating = row[0]
        reviewer_gender = row[1] 
        review_text = row[2]
        # Perform sentiment analysis
        sentiment_score = sid.polarity_scores(text=review_text)['compound']
        
        # Determine sentiment polarity
        if sentiment_score >= 0.5:
            if reviewer_gender == 'M':
                positive_reviews_male.append(review_text)
            elif reviewer_gender == 'F':
                positive_reviews_female.append(review_text)
        elif sentiment_score <= -0.5:
            if reviewer_gender == 'M':
                negative_reviews_male.append(review_text)
            elif reviewer_gender == 'F':
                negative_reviews_female.append(review_text)
        
        if number % 1000 == 0:
            print("Row", number, "done")

    print("Sentiment data processing done")
    # Output the lists of positive and negative reviews for males and females
    print("# of Positive Reviews by Males:", len(positive_reviews_male))
    print("# of Positive Reviews by Females:", len(positive_reviews_female))
    print("# of Negative Reviews by Males:", len(negative_reviews_male))
    print("# of Negative Reviews by Females:", len(negative_reviews_female))

def review_word_length():
    print("Reviewing word length")
    df = pd.DataFrame(data)
    df.columns = ["Overall_Rating", "Gender", "Review_Text"]
    
    male_reviews = df[df['Gender'] == 'M']['Review_Text']
    female_reviews = df[df['Gender'] == 'F']['Review_Text']
    avg_rev_length_male = calc_avg_review_length(male_reviews)
    avg_rev_length_female = calc_avg_review_length(female_reviews)

    # Compute average review word length for each gender
    avg_word_length_by_gender = pd.DataFrame({
        'Gender': ['Male', 'Female'],
        'Avg_Review_Length': [avg_rev_length_male, avg_rev_length_female]
    })

    # Create a boxplot to visualize the distribution of average review word length by gender
    plt.figure(figsize=(8, 6))
    plt.bar(avg_word_length_by_gender['Gender'], avg_word_length_by_gender['Avg_Review_Length'], color=['blue', 'pink'])
    plt.title('Distribution of Average Review Word Length by Gender')
    plt.xlabel('Gender')
    plt.ylabel('Average Review Length')
    plt.show()

    male_overal_rating = df[df['Gender'] == 'M']["Overall_Rating"]
    female_overal_rating = df[df['Gender'] == 'F']["Overall_Rating"]
   
    avg_male_overal_rating = male_overal_rating.mean()
    avg_female_overal_rating = female_overal_rating.mean()

    print("Calculating average review stats by gender with word length")
    # Group by gender and calculate the average review star rating and average review word length
    avg_review_stats_by_gender = pd.DataFrame({
        'Gender': ['Male', 'Female'],
        'Avg_Review_Star_Rating': [avg_male_overal_rating, avg_female_overal_rating],
        'Avg_Review_Word_Length': [avg_rev_length_male, avg_rev_length_female],
    })

    # Create a scatter plot to visualize the correlation between review star rating and average review word length by gender
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x='Avg_Review_Star_Rating', y='Avg_Review_Word_Length', hue='Gender', data=avg_review_stats_by_gender)
    plt.title('Correlation between Review Star Rating and Average Review Word Length by Gender')
    plt.xlabel('Review Star Rating')
    plt.ylabel('Average Word Length')
    plt.legend(title='Gender')
    plt.show()

def calc_avg_review_length(review_texts):
    print("Started calculating average word length")
    word_lengths = review_texts.apply(lambda text: len(text.split()))
    total_sum = word_lengths.sum()
    total_count = word_lengths.size
    avg_review_length = total_sum / total_count

    return avg_review_length


####### Add functions here #######
if __name__ == "__main__":
    visualize_distribution()
    t_test()
    diff_ratings_per_prodct()
    sentiment_analysis()
    review_word_length()
