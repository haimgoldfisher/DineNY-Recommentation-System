# Flask Restaurant Recommendation Application

## Overview

This Flask-based web application is designed to help users discover, review, and get personalized recommendations for restaurants. It integrates various frontend and backend technologies to provide a seamless and interactive user experience.

## Key Features

### User Authentication
- **Registration and Login**: Secure user registration and login functionality using unique user IDs.

### Restaurant Listings
- **Explore Restaurants**: Users can browse and explore a list of restaurants.
- **Search and Sort**: Search by restaurant name and sort by rating or number of reviews.
- **Pagination**: The restaurant list is paginated, displaying a limited number of entries per page for a smoother experience.

### User Reviews
- **Add, Update, Delete Reviews**: Users can add, update, or delete their reviews for restaurants, including providing star ratings.
- **Interactive Pop-Ups**: Reviews and detailed restaurant information are displayed in pop-up modals, allowing for easy access and interaction.

### AI-Powered Recommendations
- **Personalized Recommendations**: AI-driven recommendations based on user review history are generated and displayed in a dedicated section of the application.
- **Cluster-Based Recommendations**: The backend quickly identifies the most suitable cluster for the user based on their actions (such as adding or updating a review), and reads the corresponding recommendations from a pre-generated model. This approach ensures fast, real-time updates and enhances the user experience by delivering relevant suggestions without delay.
- **Dynamic and Static Training**: To provide the most accurate and relevant results, the system periodically trains both ALS (Alternating Least Squares) and Clustering (K-means with unique distance function) algorithms. The ALS model offers more personalized but static recommendations, while the Clustering algorithm provides less personalized but more dynamic suggestions. This training process occurs in a separate microservice, ensuring that the main application remains responsive and unaffected during these updates.

### Kafka Integration
- **Analytics and Logging**: The application integrates with Kafka to log user actions and system analytics. This feature supports real-time monitoring and provides insights for the application owner, allowing them to see if users are engaging with the AI recommendations and exploring the suggestions. This data helps improve the recommendation engine over time based on user behavior and preferences.

## Technology Stack

### Backend
- **Flask**: Manages the core backend functionalities, including routing, user authentication, and data processing.
- **Kafka**: Used for logging analytics data, monitoring user interactions, and handling background processes.
- **MongoDB**: Serves as the database, storing user information, restaurant data, reviews, and AI-generated recommendations.

### Frontend
- **HTML/CSS/JavaScript**: The frontend is built with HTML templates, styled with CSS, and made interactive using JavaScript, providing a responsive and user-friendly interface.

### Database
- **MongoDB**: A NoSQL database used to store collections like users, restaurants, reviews, and recommendations. It ensures fast data retrieval and updates.

### Pop-Up Modals
- **Interactive Pop-Ups**: Pop-ups are used for displaying restaurant details and managing reviews, ensuring a consistent and smooth user experience. Features include image consistency, star ratings, and background blurring when active.

## User Interface

- **Login and Register Pages**: Secure pages for user authentication, allowing users to register a new account or log in to an existing one. These pages ensure that only registered users can access personalized features like reviews and recommendations.
- **Home Page**: The boarding page that welcomes users to the application, providing an overview and entry point to explore the app’s features.
- **Restaurants Page**: Users can explore a comprehensive list of restaurants, with options to view more details, submit reviews, and sort or search by restaurant name, rating, or number of reviews.
- **Recommendations Page**: Showcases AI-generated restaurant recommendations tailored to the user's preferences, providing personalized suggestions based on review history.
- **Review Management**: Users can manage their reviews through dedicated pages, with intuitive modals for updating or deleting entries, ensuring easy and efficient review management.
