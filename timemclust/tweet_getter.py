import json
from datetime import datetime
import requests
from tick import hawkes


class TweetGetter:

    time_scale_denom = None
    bearer_token = None

    def __init__(self, bearer_token, time_scale_denom=(60 ** 2 * 24 * 7 * 4)):
        self.bearer_token = bearer_token
        self.time_scale_denom = time_scale_denom

    # Get the last k tweets from a user with id user_id
    def get_recent_tweets_user(self, k, user_id):
        url = "https://api.twitter.com/1.1/statuses/user_timeline.json?user_id={}&count={}".format(
            user_id, k)

        headers = {
            'Authorization': 'Bearer {}'.format(self.bearer_token),
        }

        response = requests.request("GET", url, headers=headers)
        return response.text.encode('utf8')

    def get_recent_tweets(self, k, user_ids):
        tweet_times_by_user = {}

        for user_id in user_ids:
            tweet_times = []
            recent_tweets = json.loads(self.get_recent_tweets_user(k, user_id))
            if 'errors' not in recent_tweets and 'error' not in recent_tweets:
                for recent_tweet in recent_tweets:
                    dt = datetime.strptime(
                        recent_tweet['created_at'], '%a %b %d %H:%M:%S +0000 %Y')
                    tweet_times.append(
                        (dt - datetime(1970, 1, 1)).total_seconds() / self.time_scale_denom)

            if (len(tweet_times) != 0):
                tweet_times_by_user[user_id] = tweet_times

        return (datetime.utcnow() - datetime(1970, 1, 1)).total_seconds() / self.time_scale_denom, tweet_times_by_user
