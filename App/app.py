from datetime import datetime, timezone
import random
import time
import json
from flask import Flask, render_template, request, redirect, url_for, jsonify, flash
from flask_login import LoginManager, login_user, login_required, current_user, UserMixin, logout_user
import pymongo
import logging
import os
from get_cluster import get_cluster_recommandations
from kafka import KafkaProducer
from kafka.admin import KafkaAdminClient, NewTopic

app = Flask(__name__)

# Load environment variables
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'your_secret_key')  # Default if not set
mongo_uri = os.getenv('MONGO_URI', 'mongodb://localhost:27017/')
kafka_bootstrap_servers = os.getenv('KAFKA_BOOTSTRAP_SERVERS', 'localhost:9092')
kafka_topic = os.getenv('KAFKA_TOPIC', 'analytics_topic')

# Configure logging
logging.basicConfig(level=logging.INFO)

# MongoDB configuration
mongo_client = pymongo.MongoClient(mongo_uri)
mongo_db = mongo_client['Google-Maps-Restaurant']
users_collection = mongo_db['Users']
reviews_collection = mongo_db['Reviews']
restaurants_collection = mongo_db['Restaurants']
recommendations_collection = mongo_db['Recommendations']
analytics_collection = mongo_db['Analytics']


# Function to create Kafka topic
def create_kafka_topic(topic_name, kafka_bootstrap_servers):
    admin_client = KafkaAdminClient(bootstrap_servers=kafka_bootstrap_servers)

    topic_list = [NewTopic(name=topic_name, num_partitions=1, replication_factor=1)]

    try:
        admin_client.create_topics(new_topics=topic_list, validate_only=False)
        logging.info(f"Topic '{topic_name}' created successfully.")
    except Exception as e:
        logging.warning(f"Topic '{topic_name}' might already exist or failed to create: {e}")
    finally:
        admin_client.close()


# Create Kafka topic before initializing the producer
create_kafka_topic(kafka_topic, kafka_bootstrap_servers)

# Initialize Kafka producer
producer = KafkaProducer(
    bootstrap_servers=kafka_bootstrap_servers,
    value_serializer=lambda x: json.dumps(x).encode('utf-8'),
    key_serializer=lambda k: str(k).encode('utf-8')  # Optionally serialize keys
)

login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'


# User model for Flask-Login
class User(UserMixin):
    def __init__(self, user_id):
        self.id = user_id


@login_manager.user_loader
def load_user(user_id):
    user = users_collection.find_one({"user_id": user_id})
    if user:
        return User(user_id)
    return None


def get_user_recommendations(user_id):
    user_recommendations = recommendations_collection.find_one({"user_id": user_id})
    if user_recommendations:
        return user_recommendations.get('recommendations', [])
    return []


@app.route('/')
@login_required
def index():
    return render_template('index.html')


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        user_id = request.form.get('user_id')
        if not user_id:
            flash('User ID is required.')
            return render_template('login.html')

        user = users_collection.find_one({"user_id": user_id})
        if user:
            login_user(User(user['user_id']))
            return redirect(request.args.get('next') or url_for('index'))
        flash('Invalid User ID.')

    return render_template('login.html')


def generate_unique_user_id(length=18):
    # Generate a unique user ID consisting only of digits
    while True:
        user_id = ''.join([str(random.randint(0, 9)) for _ in range(length)])
        if not users_collection.find_one({"user_id": user_id}):
            return user_id


@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        if 'confirm' in request.form:
            user_id = request.form.get('user_id')
            if not user_id or not user_id.isdigit():
                flash('A valid numeric ID is required.')
                return render_template('register.html', user_id=user_id, step='confirm')

            try:
                users_collection.insert_one({"user_id": user_id, "ratings": []})
                flash('Registration successful! You can now log in.')
                return redirect(url_for('login'))
            except Exception as e:
                logging.error(f"Error registering user: {e}")
                flash('An error occurred while registering. Please try again later.')
                return render_template('register.html', user_id=user_id, step='confirm')

        else:
            user_id = generate_unique_user_id()
            return render_template('register.html', user_id=user_id, step='confirm')

    return render_template('register.html', step='input')


@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))


@app.route('/restaurants', methods=['GET'])
@login_required
def restaurants():
    page = int(request.args.get('page', 1))
    per_page = 30
    search_query = request.args.get('search', '')
    order_by = request.args.get('order_by', 'Rating')
    sort_order = pymongo.DESCENDING if order_by in ['Rating', 'Reviews'] else pymongo.ASCENDING

    query = {"name": {"$regex": search_query, "$options": "i"}} if search_query else {}

    try:
        total = restaurants_collection.count_documents(query)
        restaurants = list(
            restaurants_collection.find(query)
            .sort(order_by, sort_order)
            .skip((page - 1) * per_page)
            .limit(per_page)
        )
    except Exception as e:
        logging.error(f"Error fetching restaurants: {e}")
        flash('An error occurred while fetching restaurants.')
        return redirect(url_for('index'))

    total_pages = (total + per_page - 1) // per_page

    return render_template(
        'restaurants.html',
        restaurants=restaurants,
        page=page,
        total_pages=total_pages,
        order_by=order_by,
        search_query=search_query
    )


@app.route('/reviews', methods=['GET'])
@login_required
def reviews():
    page = int(request.args.get('page', 1))
    per_page = 30
    user_id = current_user.id

    try:
        # Fetch user document
        user_doc = users_collection.find_one({"user_id": user_id}, {"ratings": 1})

        if not user_doc or 'ratings' not in user_doc:
            logging.info(f"No reviews found for user_id: {user_id}")
            user_reviews = []
            total = 0
        else:
            # Get all user reviews
            user_reviews = user_doc['ratings']
            total = len(user_reviews)

            # Paginate user reviews
            start = (page - 1) * per_page
            end = start + per_page
            user_reviews = user_reviews[start:end]

            # Extract gmap_ids from the user reviews
            gmap_ids = [review['gmap_id'] for review in user_reviews]

            # Fetch restaurant details for the gmap_ids
            restaurants = list(
                restaurants_collection.find({"gmap_id": {"$in": gmap_ids}})
            )

            # Create a dictionary for quick lookup of restaurant details by gmap_id
            restaurant_dict = {restaurant['gmap_id']: restaurant for restaurant in restaurants}

            # Add img_url and name to each review
            for review in user_reviews:
                gmap_id = review['gmap_id']
                if gmap_id in restaurant_dict:
                    review['img_url'] = restaurant_dict[gmap_id].get('img_url')
                    review['name'] = restaurant_dict[gmap_id].get('name')

    except Exception as e:
        logging.error(f"Error fetching reviews: {e}")
        flash('An error occurred while fetching reviews.')
        return redirect(url_for('index'))

    total_pages = (total + per_page - 1) // per_page

    return render_template(
        'reviews.html',
        reviews=user_reviews,
        page=page,
        total_pages=total_pages
    )


def update_ai_recommendations(user_id):
    # Fetch user reviews
    user_reviews = users_collection.find_one({"user_id": user_id}, {"ratings": 1})

    if not user_reviews or not user_reviews.get('ratings'):
        # No reviews for the user, delete existing recommendations and return empty list
        recommendations_collection.delete_one({"user_id": user_id})
        return []

    # Get the list of gmap_ids that the user has reviewed
    reviewed_gmap_ids = {review['gmap_id'] for review in user_reviews['ratings']}

    # Fetch the top 13 recommendations from the cluster recommendations
    recommendations = get_cluster_recommandations(user_id)

    # Filter out recommendations that the user has already reviewed
    filtered_recommendations = [
                                   rec for rec in recommendations
                                   if rec['gmap_id'] not in reviewed_gmap_ids
                               ][:13]  # Limit to top 13 recommendations

    # Update or insert recommendations for the user
    recommendations_collection.update_one(
        {"user_id": user_id},
        {"$set": {"recommendations": filtered_recommendations}},
        upsert=True
    )
    return filtered_recommendations


def update_restaurant_stats(gmap_id):
    reviews = list(reviews_collection.find({"gmap_id": gmap_id}))
    num_reviews = len(reviews)
    avg_rating = sum(review['rating'] for review in reviews) / num_reviews if num_reviews > 0 else 0

    # Log the recalculated statistics
    logging.info(
        f"Updating stats for gmap_id: {gmap_id}. Number of reviews: {num_reviews}, Average rating: {avg_rating}")

    restaurants_collection.update_one(
        {"gmap_id": gmap_id},
        {"$set": {"Rating": round(avg_rating, 1), "Reviews": num_reviews}}
    )


def update_user_reviews(user_id, gmap_id, rating=None, insert=True):
    try:
        if insert:
            review = {
                "rating": rating,
                "timestamp": int(time.time() * 1000),  # current timestamp in milliseconds
                "gmap_id": gmap_id
            }

            # Check if the review already exists
            user = users_collection.find_one({"user_id": user_id, "ratings.gmap_id": gmap_id})
            if user:
                logging.info(f"Existing review found for user_id: {user_id}, gmap_id: {gmap_id}. Updating review.")
                # Update the existing review
                result = users_collection.update_one(
                    {"user_id": user_id, "ratings.gmap_id": gmap_id},
                    {"$set": {"ratings.$.rating": rating, "ratings.$.timestamp": review["timestamp"]}}
                )
                logging.info(f"Update result: {result.raw_result}")
            else:
                logging.info(f"No existing review found for user_id: {user_id}, gmap_id: {gmap_id}. Adding new review.")
                # Add the new review to the user's reviews
                result = users_collection.update_one(
                    {"user_id": user_id},
                    {"$push": {"ratings": review}}
                )
                logging.info(f"Insert result: {result.raw_result}")
        else:
            logging.info(f"Deleting review for user_id: {user_id}, gmap_id: {gmap_id}.")
            # Delete the review
            result = users_collection.update_one(
                {"user_id": user_id},
                {"$pull": {"ratings": {"gmap_id": gmap_id}}}
            )
            logging.info(f"Delete result: {result.raw_result}")
        try:
            update_ai_recommendations(user_id)
        except Exception as e:
            logging.error(f"Error creating new AI recommendations: {e}")
            return False
        return True
    except Exception as e:
        logging.error(f"Error updating user reviews: {e}")
        return False


@app.route('/api/add_review', methods=['POST'])
@login_required
def add_review():
    user_id = current_user.id
    gmap_id = request.form.get('gmap_id')
    rating = request.form.get('rating')

    if not gmap_id or not rating:
        return jsonify({"success": False, "error": "Missing gmap_id or rating"}), 400

    try:
        rating = int(rating)
        if rating < 1 or rating > 5:
            return jsonify({"success": False, "error": "Rating must be between 0 and 5"}), 400
    except ValueError:
        return jsonify({"success": False, "error": "Invalid rating value. Rating must be a number."}), 400

    try:
        reviews_collection.update_one(
            {"user_id": user_id, "gmap_id": gmap_id},
            {"$set": {"rating": rating}},
            upsert=True
        )

        if update_user_reviews(user_id, gmap_id, rating):
            update_restaurant_stats(gmap_id)
            try:
                update_ai_recommendations(user_id)
            except Exception as e:
                logging.error(f"Error creating new AI recommendations: {e}")
                return jsonify(
                    {"success": False, "error": "An error occurred while creating new AI recommendations."}), 500
            return jsonify({"success": True})
        else:
            return jsonify({"success": False, "error": "An error occurred while updating the user reviews."}), 500
    except Exception as e:
        logging.error(f"Error adding review: {e}")
        return jsonify(
            {"success": False, "error": "An error occurred while adding the review. Please try again later."}), 500


@app.route('/api/update_review', methods=['POST'])
@login_required
def update_review():
    user_id = current_user.id
    gmap_id = request.form.get('gmap_id')
    rating = request.form.get('rating')

    if not gmap_id or not rating:
        return jsonify({"success": False, "error": "Missing gmap_id or rating"}), 400

    try:
        rating = int(rating)
        if rating < 1 or rating > 5:
            return jsonify({"success": False, "error": "Rating must be between 0 and 5"}), 400
    except ValueError:
        return jsonify({"success": False, "error": "Invalid rating value. Rating must be a number."}), 400

    try:
        reviews_collection.update_one(
            {"user_id": user_id, "gmap_id": gmap_id},
            {"$set": {"rating": rating}}
        )

        if update_user_reviews(user_id, gmap_id, rating):
            update_restaurant_stats(gmap_id)
            return jsonify({"success": True})
        else:
            return jsonify({"success": False, "error": "An error occurred while updating the user reviews."}), 500
    except Exception as e:
        logging.error(f"Error updating review: {e}")
        return jsonify({"success": False, "error": "An error occurred while updating the review"}), 500


@app.route('/api/delete_review', methods=['POST'])
@login_required
def delete_review():
    user_id = current_user.id
    gmap_id = request.form.get('gmap_id')

    if not gmap_id:
        return jsonify({"success": False, "error": "Missing gmap_id"}), 400

    try:
        # Delete review from the reviews collection
        reviews_collection.delete_one({"user_id": user_id, "gmap_id": gmap_id})

        # Remove review from the users collection
        users_collection.update_one(
            {"_id": user_id},
            {"$pull": {"ratings": {"gmap_id": gmap_id}}}
        )

        logging.info(f"Review deleted for user_id: {user_id}, gmap_id: {gmap_id}")
        if update_user_reviews(user_id, gmap_id, None, False):
            update_restaurant_stats(gmap_id)
            return jsonify({"success": True})

    except Exception as e:
        logging.error(f"Error deleting review: {e}")
        return jsonify(
            {"success": False, "error": "An error occurred while deleting the review. Please try again later."}), 500


@app.route('/ai_recommendations')
@login_required
def ai_recommendations():
    user_id = current_user.id
    user_recommendations = recommendations_collection.find_one({'user_id': user_id})
    if not user_recommendations:
        logging.debug(f"No recommendations found for user_id: {user_id}")
        return render_template('ai_recommendations.html', recommendations=[])

    recommendations = user_recommendations.get('recommendations', [])
    rec_ids = [rec.get('gmap_id') for rec in recommendations]
    # Fetch restaurant details based on gmap_ids
    restaurants = list(restaurants_collection.find({'gmap_id': {'$in': rec_ids}}))
    # Create a dictionary for quick lookup
    restaurant_dict = {rest['gmap_id']: rest for rest in restaurants}
    # Enrich recommendations with restaurant details and pre-fetched images
    enriched_recommendations = [
        {
            'name': restaurant_dict.get(rec.get('gmap_id', ''), {}).get('name', 'Name not available'),
            'gmap_id': rec.get('gmap_id', ''),
            'address': restaurant_dict.get(rec.get('gmap_id', ''), {}).get('address', 'Address not available'),
            'url': restaurant_dict.get(rec.get('gmap_id', ''), {}).get('url', 'URL not available'),
            'prediction': rec.get('score', 0),
            'avg_rating': restaurant_dict.get(rec.get('gmap_id', ''), {}).get('Rating', 0),
            'num_reviews': restaurant_dict.get(rec.get('gmap_id', ''), {}).get('Reviews', 0),
            'img_url': restaurant_dict.get(rec.get('gmap_id', ''), {}).get('img_url',
                                                                           'https://example.com/default_image.jpg')
        }
        for rec in recommendations
    ]
    return render_template('ai_recommendations.html', recommendations=enriched_recommendations[:10])


@app.route('/restaurant_details/<gmap_id>', methods=['GET'])
def restaurant_details(gmap_id):
    app.logger.info(f"Fetching details for gmap_id: {gmap_id}")
    # Fetch restaurant details from MongoDB
    restaurant = restaurants_collection.find_one({'gmap_id': gmap_id})
    if restaurant:
        # Construct response DB
        data = {
            'name': restaurant['name'],
            'img_url': restaurant['img_url'],
            'rating': restaurant.get('Rating', 'N/A'),
            'reviews': restaurant.get('Reviews', 0),
            'address': restaurant.get('address', 'No Address available.'),
            'url': restaurant.get('url', 'No URL available.')
        }
        return jsonify(data)
    else:
        return jsonify({'error': 'Restaurant not found'}), 404


@app.route('/review_details/<gmap_id>', methods=['GET'])
def review_details(gmap_id):
    # Fetch restaurant details from MongoDB
    user_id = current_user.id
    restaurant = restaurants_collection.find_one({'gmap_id': gmap_id})
    review = reviews_collection.find_one({'user_id': user_id, 'gmap_id': gmap_id})

    if restaurant:
        # Construct response DB
        data = {
            'name': restaurant.get('name', 'No Name Available'),
            'img_url': restaurant.get('img_url', 'No Image Available'),
            'avg_rating': restaurant.get('Rating', 'N/A'),
            'user_rating': review.get('Rating', 'N/A') if review else 'No Rating Available',
            'reviews': restaurant.get('Reviews', 0),
            'address': restaurant.get('address', 'No Address Available'),
            'url': restaurant.get('url', 'No URL Available')
        }
        return jsonify(data)
    else:
        return jsonify({'error': 'Restaurant not found'}), 404


@app.route('/enter', methods=['GET'])
@login_required
def user_entered():
    user_id = current_user.id
    # ip_address = request.remote_addr
    entering_time = datetime.now(timezone.utc).isoformat()

    event = {
        'event_type': 'user_entered',
        'user_id': user_id,
        # 'ip': ip_address,
        'entering_time': entering_time
    }

    # Send message to Kafka
    producer.send(kafka_topic, key=str(user_id), value=event)
    producer.flush()

    return redirect(url_for('index'))


@app.route('/click/<gmap_id>', methods=['POST'])
@login_required
def ai_recommendation_click(gmap_id):
    user_id = current_user.id

    event = {
        'event_type': 'url_click',
        'user_id': user_id,
        'gmap_id': gmap_id
    }

    # Send message to Kafka
    producer.send(kafka_topic, key=str(user_id), value=event)
    producer.flush()

    return redirect(url_for('ai_recommendations'))


@app.route('/exit', methods=['POST'])
@login_required
def user_exited():
    user_id = current_user.id
    exiting_time = datetime.now(timezone.utc).isoformat()

    event = {
        'event_type': 'user_exited',
        'user_id': user_id,
        'exiting_time': exiting_time
    }

    # Send message to Kafka
    producer.send(kafka_topic, key=str(user_id), value=event)
    producer.flush()

    return redirect(url_for('login'))


@app.route('/events', methods=['POST'])
def receive_event():
    try:
        event = request.json
        analytics_collection.insert_one(event)
        return jsonify({"status": "success"}), 200
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5050)
